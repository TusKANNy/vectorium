### Quantization

We have a $d$-dimesional dataset of $n$ elements $ \mathcal{X}$ and we want to quantize every $x_i \in \mathcal{X}$, with $x_i \in \mathbb{R}^d$. 

We first compute the centroid of the dataset $c \in \mathbb{R}^d$. The $j$-th component of $c$, namely $c_j = \frac{1}{n}\sum_{i=0}^{n-1} x_{i,j}   $. Namely the means of each component.

We center $x$ by subtracting $c$ to it, thus obtaining 
$$ \bar{x}_i = x_i - c $$

We randomly rotate $\bar{x}_i$ by multiplying it by an orthonormal matrix $P$, obtaining the rotate residual (I think we would actually multiply by $P^{-1}$, but this is an operative description.).

$$ r_i = P \cdot \bar{x}_i$$


We binarize the vector $r_i$ obtaining $o_i \in \{-1, 1\}^D$. There is a $\frac{1}{\sqrt D}$ factor that is never materialized (it is folded into the per-query `scale`).

$$o_{i,j} = \text{sign}(r_{i,j}) \cdot \frac{1}{\sqrt D} $$

To mitigate the loss of information given by binarization, we store $||r_i||_2$ and the factor $\bar f = \frac{\sum_j |r_{i,j}|}{\sqrt D\, ||r_i||_2}$ (the cosine between the code and the true residual). They are not stored raw: each document is $D/64$ code words plus **one** metadata word packing the two scan-ready constants, $\texttt{f\_add} = ||r_i||^2$ and $\texttt{s} = ||r_i|| / \bar f$, so the scan needs no divide (`pack_metadata`).

When we need to compute the distance with the query, if the distance is euclidean, we center the query $q_c = q - c$, as euclidean distance is translation invariant. Centering the query a MIPS problem (dot product) does not maintain the order, so we have to compute

$$ \langle q, x_i\rangle = \langle q, c + P^{\top} r_i \rangle =   \langle c, q \rangle + \langle r_i, P q \rangle $$

where $\langle c, q\rangle$ is computed once per query together with the query rotation for the dot product. 

Then it does a complex quantization of the query to obtain
$\bar{q}$, where
$$\bar{q}_i = \Delta c_i + vl  $$

where $c_i$ is chosen as to minimize $ ||q - \bar{q}||_2$, with $c_i \in [0, 2^{B}-1]$, where $B$ is the number of bits assigned to the query. 

Each code of the query $c_j$ is decomposed into its bit planes

$$ c_j = \sum_{b=0}^{B-1} c^{(b)}_j \cdot 2^{b} $$

where $c^{(b)}$ is the bit of weight $2^b$ of $c_j$ (least significant first, $b=0$ is the LSB).
Example: $c_j = 9$, $B=4$ is $1001$, so $c^{(0)}=1, c^{(1)}=c^{(2)}=0, c^{(3)}=1$, and $1 + 8 = 9$.

Planes are stored plane-major (`pack_bit_planes`), so each plane is a contiguous $D/64$-word slice that the kernel can AND against the document code in one streaming pass.

so the actual quantity we compute, set aside the correction terms, is the agreement between the document sign code $o_i$ and the reconstructed query $\bar{q}$. Writing the stored bit $b_{i,j} = [\, r_{i,j} \ge 0 \,] \in \{0,1\}$ and its sign $s_{i,j} = 2 b_{i,j} - 1 = \text{sign}(r_{i,j})$, and using the query bit-plane decomposition above, this is a sum of masked popcounts:

$$ \langle b_i, c \rangle = \sum_{b=0}^{B-1} 2^{b}\ \text{popcount}\big( b_i \wedge c^{(b)} \big) $$

$$ \langle s_i, \bar{q} \rangle = 2\big( \Delta\, \langle b_i, c \rangle + vl \cdot \text{popcount}(b_i) \big) - \big( \Delta \textstyle\sum_j c_j + vl\, D \big) $$

We turn this into an estimate of $\langle r_i, r_q \rangle$ (where $r_q$ is the rotated query and $\|r_q\|$ its norm) by normalizing to the unit code, debiasing with the stored factor $\bar f$, and rescaling by the two norms:

$$ \overline{ip} = \frac{\langle s_i, \bar{q}\rangle}{\sqrt D\, \|r_q\|}, \qquad \cos \approx \frac{\overline{ip}}{\bar{f}}, \qquad \boxed{\ \langle r_i, r_q\rangle \approx \|r_i\|\, \|r_q\|\, \cos\ } $$

For a 1-bit query the query code is itself unit-norm, so $\|r_q\|$ drops out of the agreement term and the normalization is by $D$ instead:

$$ \overline{ip} = \frac{D - 2\,\text{hamming}(b_i, b_q)}{D} $$

In code both cases are `term = s · ip_raw · scale` (`compute_distance`), with `s` from the metadata word and the per-query `scale` absorbing the rest: `ip_raw` $= D - 2\,\text{hamming}$ with `scale` $=\|r_q\|/D$ for 1 bit, `ip_raw` $= \langle s_i,\bar q\rangle$ with `scale` $= 1/\sqrt D$ for multi-bit.

### The two paths: dot product and Euclidean

The shared estimate $\langle r_i, r_q\rangle$ enters a different final combine — and needs a different query preprocessing $r_q$ — for each metric.

**Dot product (MIPS).** The query is rotated but **not** centered, $r_q = P q$, because centering does not preserve the inner-product order. Since $P$ is orthogonal, $x_i = c + P^{\top} r_i$, so the centroid term is added back in closed form:

$$ \langle x_i, q\rangle = \langle c, q\rangle + \langle r_i, P q\rangle \approx \underbrace{\langle c, q\rangle}_{\text{per-query const.}} + \ \|r_i\|\, \|r_q\|\, \cos $$

$\langle c, q\rangle$ is computed once per query, together with the query rotation $P q$. Larger = nearer.

**Euclidean distance.** Distance is translation-invariant, so we center the query with the same centroid, $r_q = P(q - c)$; the centroid then cancels exactly ($\|x_i - q\|^2 = \|\bar{x}_i - (q - c)\|^2 = \|r_i - r_q\|^2$):

$$ \|x_i - q\|^2 = \|r_i\|^2 + \|r_q\|^2 - 2\langle r_i, r_q\rangle \approx \|r_i\|^2 + \|r_q\|^2 - 2\,\|r_i\|\, \|r_q\|\, \cos $$

$\|r_i\|^2$ is stored per document and $\|r_q\|^2$ is a per-query constant. Smaller = nearer.








