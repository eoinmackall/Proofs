[
  {
    "id": "ref_1",
    "slogan": "Double centralizer theorem for central simple algebras, including the centralizer of a subfield and the maximal subfield criterion.",
    "formal statement": "Let A be a central simple k-algebra and let B be a simple k-subalgebra of A. Then the centralizer C_A(B) is simple, C_A(C_A(B)) = B, and dim_k(B) * dim_k(C_A(B)) = dim_k(A). If K is a subfield of A containing k, then C_A(K) is a central simple K-algebra of degree deg(A)/[K:k] that is Brauer equivalent to A tensor_k K. In particular, K is a maximal subfield of A if and only if C_A(K) = K.",
    "reference": "Theorem 1",
    "tags": [
      "central simple algebras",
      "centralizers",
      "brauer equivalence",
      "maximal subfields"
    ]
  },
  {
    "id": "ref_2",
    "slogan": "Structure theorem for G-Galois algebras and for maximal commutative etale subalgebras of central simple algebras.",
    "formal statement": "Let G be a finite group and let L be a commutative etale k-algebra of dimension |G| with an action of G by k-algebra automorphisms such that L^G = k. (i) G permutes the primitive idempotents e_1,...,e_m of L transitively. If H is the stabilizer of e_1, then M = L e_1 is a field, H acts faithfully on M, M/k is Galois with group H, m = [G:H], and L is isomorphic to M^m as a k-algebra; in particular L is a field if and only if H = G, and then L/k is a Galois field extension with group G. (ii) For sigma != tau in G, the elements sigma(ell) - tau(ell) with ell in L generate the unit ideal of L. (iii) Let A be a central simple k-algebra of degree n and let L' be a commutative etale subalgebra of A with dim_k(L') = n. Then C_A(L') = L', and dim_k(A e) = n dim_k(L' e) for every idempotent e of L'. (iv) In the situation of (iii), every k-algebra automorphism alpha of L' is induced by an inner automorphism of A: there is u in A^\u00d7 with u ell u^{-1} = alpha(ell) for all ell in L'. (v) Conversely, if M/k is a Galois field extension with group H and H is a subgroup of G, then Ind_H^G M = { f : G -> M : f(hg) = h(f(g)) for all h in H, g in G }, with pointwise operations and action (sigma f)(g) = f(g sigma), is a G-Galois algebra, and as a k-algebra it is isomorphic to M^{[G:H]}.",
    "reference": "Theorem 2A",
    "tags": [
      "galois algebras",
      "central simple algebras",
      "etale algebras",
      "centralizers",
      "induction"
    ]
  },
  {
    "id": "ref_3",
    "slogan": "Crossed product criterion: central simple algebras with G-Galois maximal commutative subalgebras are exactly G-crossed products built from normalized 2-cocycles.",
    "formal statement": "Let A be a central simple k-algebra of degree n, let G be a group of order n, and let L be a commutative subalgebra of A with dim_k(L)=n that is a G-Galois algebra (commutative etale of dimension |G| with G-action and L^G=k). Then there are units u_sigma in A^\u00d7 for sigma in G, with u_1=1, such that u_sigma ell u_sigma^{-1}=sigma(ell) for all ell in L, A = direct sum_{sigma in G} L u_sigma, and u_sigma u_tau = f(sigma,tau) u_{sigma tau} for a normalized 2-cocycle f : G x G -> L^\u00d7. Conversely, for every G-Galois algebra L and every normalized 2-cocycle f : G x G -> L^\u00d7, the algebra (L,G,f) = direct sum_{sigma in G} L u_sigma with these relations is central simple of degree n and contains L as a maximal commutative subalgebra. When L is a field, this is the classical crossed product.",
    "reference": "Theorem 2",
    "tags": [
      "crossed products",
      "central simple algebras",
      "galois algebras",
      "cocycles"
    ]
  },
  {
    "id": "ref_4",
    "slogan": "Galois splitting field criterion for G-crossed products, with consequences for division algebras, cyclicity, matrix algebras, and induction.",
    "formal statement": "Let A be a central simple k-algebra of degree n and let G be a group of order n. Then A is a G-crossed product (i.e. contains a commutative subalgebra L of dimension n that is a G-Galois algebra) if and only if there are a subgroup H of G and a Galois field extension M/k with Gal(M/k) isomorphic to H such that M splits A. In that case A is isomorphic to M_{[G:H]}(B), where B is Brauer equivalent to A, has degree |H|, and contains M as a maximal subfield. Consequently: (i) a division algebra A is a G-crossed product if and only if it contains a maximal subfield that is Galois over k with group G; (ii) A is cyclic if and only if A is split by a cyclic field extension of k whose degree divides n; (iii) M_n(k) is a G-crossed product for every group G of order n, in particular it is cyclic; (iv) if B is an H-crossed product and H is a subgroup of G, then M_{[G:H]}(B) is a G-crossed product.",
    "reference": "Theorem 2B",
    "tags": [
      "crossed products",
      "galois extensions",
      "splitting fields",
      "central simple algebras"
    ]
  },
  {
    "id": "ref_5",
    "slogan": "Embedding criterion for subfields of central simple algebras, with the maximal subfield case as splitting.",
    "formal statement": "Let A be a central simple k-algebra of degree n and let K/k be a field extension of degree m with m dividing n. Then K embeds in A as a k-subalgebra if and only if A tensor_k K is isomorphic to M_m(B) for some central simple K-algebra B, equivalently ind(A tensor_k K) divides n/m. In particular, when m=n, K embeds in A as a maximal subfield if and only if K splits A.",
    "reference": "Theorem 3",
    "tags": [
      "embedding criteria",
      "central simple algebras",
      "maximal subfields",
      "index"
    ]
  },
  {
    "id": "ref_6",
    "slogan": "Frobenius base change multiplies Brauer classes by p and gives purely inseparable splitting for p-power exponent.",
    "formal statement": "Let char(k)=p>0 and let A be a central simple k-algebra. Under the ring isomorphism phi : k^{1/p} -> k, x -> x^p, the class [A tensor_k k^{1/p}] in Br(k^{1/p}) is carried to p[A] in Br(k). Consequently ind(A tensor_k k^{1/p}) = ind(A^{\u2297 p}), and if exp(A)=p^e then A is split by k^{1/p^e}, hence by a finite purely inseparable extension F with F^{p^e} subset k.",
    "reference": "Theorem 4",
    "tags": [
      "brauer group",
      "frobenius",
      "purely inseparable extensions",
      "index"
    ]
  },
  {
    "id": "ref_7",
    "slogan": "Albert theorem: a degree p division algebra with a p-central element is cyclic and has an Artin-Schreier maximal subfield.",
    "formal statement": "Let char(k)=p and let D be a division algebra of degree p with center k. Suppose D contains a p-central element x, meaning x notin k and x^p=b in k. Then D contains an Artin-Schreier maximal subfield k(u) with u^p-u=w in k and x u x^{-1}=u+1. Hence D is isomorphic to the symbol algebra [w,b), the k-algebra generated by u and v with u^p-u=w, v^p=b, and v u v^{-1}=u+1, and D is cyclic.",
    "reference": "Theorem 6",
    "tags": [
      "division algebras",
      "artin-schreier extensions",
      "cyclic algebras",
      "p-central elements"
    ]
  },
  {
    "id": "ref_8",
    "slogan": "Witt theorem: H^2 with Z/p coefficients vanishes and cyclic p-extensions embed in cyclic extensions of one higher p-power degree.",
    "formal statement": "Let char(k)=p. Then H^2(G_k, Z/p) = 0, and every cyclic extension L/k of degree p^e is contained in a cyclic extension M/k of degree p^{e+1}.",
    "reference": "Theorem 7",
    "tags": [
      "galois cohomology",
      "artin-schreier theory",
      "cyclic extensions"
    ]
  },
  {
    "id": "ref_9",
    "slogan": "Reduction of degree p^2 crossed product questions to the division algebra case.",
    "formal statement": "Let char(k)=p and let A be a central simple k-algebra of degree p^2 that is not a division algebra. Write A = M_r(D) with D a division algebra, so deg(D) is 1 or p. (i) If D=k, or if deg(D)=p and D is cyclic (automatic when p=2 or 3), then A is a G-crossed product for every group G of order p^2; in particular A is cyclic. (ii) Suppose some cyclic field extension L/k of degree p splits D. Then A contains a maximal subfield that is a cyclic field extension of k of degree p^2. (iii) If deg(D)=p and D is not cyclic, then A is a crossed product if and only if D is split by a Galois field extension of degree p^2.",
    "reference": "Theorem 8",
    "tags": [
      "p-algebras",
      "crossed products",
      "division algebras",
      "cyclic extensions"
    ]
  },
  {
    "id": "ref_10",
    "slogan": "Tensor products of crossed products are crossed products, with Galois group the product of the groups.",
    "formal statement": "Let A_1 and A_2 be central simple k-algebras that are G_1- and G_2-crossed products, with G_i-Galois algebras L_i contained in A_i. Then L_1 tensor_k L_2, contained in A_1 tensor_k A_2, is a (G_1 x G_2)-Galois algebra, so A_1 tensor_k A_2 is a (G_1 x G_2)-crossed product. If L_1 tensor_k L_2 is a field, it is a maximal subfield of A_1 tensor_k A_2 that is Galois over k with group G_1 x G_2.",
    "reference": "Theorem 9",
    "tags": [
      "crossed products",
      "tensor products",
      "galois algebras"
    ]
  },
  {
    "id": "ref_11",
    "slogan": "Tensor products of two cyclic degree p algebras are crossed products with group Z/p x Z/p, and division cases have Galois maximal subfields.",
    "formal statement": "Let char(k)=p and let A be isomorphic to D_1 tensor_k D_2, where D_1 and D_2 are cyclic central simple k-algebras of degree p. Then A is a crossed product with group Z/p x Z/p. If A is a division algebra, then A contains a maximal subfield that is Galois over k with group Z/p x Z/p.",
    "reference": "Theorem 10",
    "tags": [
      "p-algebras",
      "tensor products",
      "crossed products",
      "galois maximal subfields"
    ]
  },
  {
    "id": "ref_12",
    "slogan": "Index divisibility under finite field extensions, with the degree p case characterized by embedding.",
    "formal statement": "Let D be a central division k-algebra and let K/k be a finite field extension. Then ind(D tensor_k K) divides ind(D), and ind(D) divides [K:k] * ind(D tensor_k K). In particular, if ind(D)=p^n and [K:k]=p, then ind(D tensor_k K) is either p^n or p^{n-1}, and it equals p^{n-1} if and only if K embeds in D.",
    "reference": "Theorem 11",
    "tags": [
      "index",
      "division algebras",
      "field extensions",
      "embedding criteria"
    ]
  },
  {
    "id": "ref_13",
    "slogan": "Artin-Schreier pairs inside division p-algebras: p-central elements produce cyclic degree p Artin-Schreier subextensions.",
    "formal statement": "Let char(k)=p, let D be a central division k-algebra, and let x in D not in k satisfy x^p in k. Then there exists u in D with x u x^{-1} = u + 1. For any such u, put w = u^p - u. Then w commutes with x and u, the extension k(u)/k(w) is a cyclic Artin-Schreier extension of degree p, and conjugation by x restricts to a generator of Gal(k(u)/k(w)).",
    "reference": "Theorem 12",
    "tags": [
      "p-central elements",
      "artin-schreier extensions",
      "division algebras"
    ]
  },
  {
    "id": "ref_14",
    "slogan": "Artin-Schreier theory: degree p Galois extensions are classified by K/wp(K) and their equality by F_p-lines.",
    "formal statement": "Let K be a field of characteristic p and let wp(t)=t^p-t. Then H^1(G_K, Z/p) is isomorphic to K/wp(K). Consequently: (a) every Galois extension of K of degree p has the form K(wp^{-1}a) with a notin wp(K), where K(wp^{-1}a)=K(theta) for theta^p-theta=a; (b) for a,b notin wp(K), the fields K(wp^{-1}a) and K(wp^{-1}b) coincide inside a separable closure K_s if and only if b is in F_p^\u00d7 a + wp(K).",
    "reference": "Theorem 13",
    "tags": [
      "artin-schreier theory",
      "galois cohomology",
      "degree p extensions"
    ]
  },
  {
    "id": "ref_15",
    "slogan": "Characterization of Galois maximal subfields in degree p^2 central simple algebras by Artin-Schreier data in a cyclic subfield.",
    "formal statement": "Let char(k)=p and let A be a central simple k-algebra of degree p^2. Then A contains a maximal subfield that is Galois over k if and only if there exist a subfield K of A that is cyclic of degree p over k, with generator sigma of Gal(K/k), and an element w in K notin wp(K) such that: (i) C_A(K) contains an element u with u^p-u=w; and (ii) sigma(w) is in F_p^\u00d7 w + wp(K). In that case K(u) is a maximal subfield of A that is Galois over k. If A is a division algebra, this condition is equivalent to A being a crossed product. Moreover, if A is a division algebra, then C_A(K) is a division algebra of degree p over K; if in addition C_A(K) is cyclic over K, it has the form [w',b)_K for some w' in K and b in K^\u00d7.",
    "reference": "Theorem 14",
    "tags": [
      "galois maximal subfields",
      "p-algebras",
      "centralizers",
      "crossed products"
    ]
  },
  {
    "id": "ref_16",
    "slogan": "Noether-Kothe theorem: every central division algebra has a separable maximal subfield.",
    "formal statement": "Every central division k-algebra D contains a maximal subfield that is separable over k.",
    "reference": "Theorem 15",
    "tags": [
      "division algebras",
      "separable extensions",
      "maximal subfields"
    ]
  },
  {
    "id": "ref_17",
    "slogan": "Calculus of p-symbol algebras: additivity in the Artin-Schreier and norm parameters and splitting criterion.",
    "formal statement": "Let char(k)=p, let a,a' in k and b,b' in k^\u00d7. The symbol algebra [a,b), generated by u and v with u^p-u=a, v^p=b, and v u v^{-1}=u+1, is a central simple k-algebra of degree p. It is cyclic, with the Z/p-Galois algebra L_a=k[u], which is a field if and only if a notin wp(k). In Br(k): (i) [a,b) + [a',b) = [a+a', b); (ii) [a,b) + [a,b') = [a, bb'); (iii) [a,b) is split if a in wp(k); if a notin wp(k), then [a,b) is split if and only if b is a norm from k(wp^{-1}a).",
    "reference": "Theorem 16",
    "tags": [
      "p-symbols",
      "brauer group",
      "cyclic algebras",
      "norms"
    ]
  },
  {
    "id": "ref_18",
    "slogan": "Albert-Teichmuller theorem: p-algebras are Brauer equivalent to cyclic algebras and p-torsion is generated by p-symbols.",
    "formal statement": "Let char(k)=p. Every p-algebra over k (a central simple k-algebra whose degree is a power of p) is Brauer equivalent to a cyclic algebra. Moreover, the p-torsion subgroup Br(k)[p] is generated by the classes of the symbol algebras [a,b).",
    "reference": "Theorem 17",
    "tags": [
      "p-algebras",
      "brauer group",
      "cyclic algebras",
      "symbol algebras"
    ]
  },
  {
    "id": "ref_19",
    "slogan": "Albert cyclicity criterion: degree p^n division algebras are cyclic exactly when they have a simple purely inseparable maximal subfield.",
    "formal statement": "Let char(k)=p and let D be a central division k-algebra of degree p^n. Then D is cyclic if and only if D contains a simple purely inseparable maximal subfield, that is, a subfield k(x) with [k(x):k]=p^n and x^{p^n} in k. For x in D with x^{p^n} in k, the following are equivalent: [k(x):k]=p^n; x^{p^n} notin k^p; x^{p^{n-1}} notin k. The criterion does not extend to split algebras: M_{p^n}(k) is cyclic for every k, but over a perfect field such as F_p there is no x with x^{p^n} in k and x^{p^n} notin k^p.",
    "reference": "Theorem 18",
    "tags": [
      "cyclicity criteria",
      "purely inseparable extensions",
      "division algebras",
      "p-algebras"
    ]
  },
  {
    "id": "ref_20",
    "slogan": "Amitsur-Saltman existence of noncyclic division p-algebras of degree p^2.",
    "formal statement": "For every prime p there exist fields k of characteristic p and division p-algebras over k of degree p^2 that are not cyclic. The examples are generic abelian crossed products with group Z/p x Z/p, and they contain no p-central element.",
    "reference": "Theorem 19",
    "tags": [
      "noncyclic p-algebras",
      "generic crossed products",
      "p-central elements"
    ]
  },
  {
    "id": "ref_21",
    "slogan": "Artin-Schreier-Witt theory: Witt vector symbols give cyclic Galois algebras and classify H^1 with Z/p^m coefficients.",
    "formal statement": "Let char(k)=p and m>=1. For omega in W_m(k), the ring of Witt vectors of length m over k, let L_omega be the k-algebra generated by the coordinates of a solution x=(x_0,...,x_{m-1}) to F(x)-x=omega, where F is the Frobenius, with automorphism sigma(x)=x+(1,0,...,0). Then L_omega is a cyclic Galois algebra of degree p^m with generator sigma. Its character chi_omega in H^1(G_k, Z/p^m) is given by chi_omega(gamma)=gamma(xi)-xi in W_m(F_p)=Z/p^m for any solution xi in W_m(k_s) of F(xi)-xi=omega, where k_s is a separable closure of k. The map omega -> chi_omega induces an isomorphism W_m(k)/wp(W_m(k)) \u2245 H^1(G_k, Z/p^m), where wp=F-1. Moreover: (i) L_omega is a field if and only if omega_0 notin wp(k). For m=2 and w=(w_0,w_1) with w_0 notin wp(k), the unique subfield of L_w of degree p is k(wp^{-1}w_0), on which sigma restricts to x_0 -> x_0+1. (ii) For omega in W_m(k), chi_{V(omega)} = iota \u2218 chi_omega, where V is the Verschiebung and iota : Z/p^m -> Z/p^{m+1} is multiplication by p; under the standard embeddings into Q/Z, chi_{V(omega)} = chi_omega.",
    "reference": "Theorem 20",
    "tags": [
      "witt vectors",
      "artin-schreier-witt theory",
      "galois cohomology",
      "cyclic galois algebras"
    ]
  },
  {
    "id": "ref_22",
    "slogan": "Calculus of cyclic algebras: classification by character and parameter, additivity, matrix decomposition, and Witt symbol consequences.",
    "formal statement": "Let L be a cyclic Galois algebra of degree n with generator sigma, and let b,b' in k^\u00d7, where (L,sigma,b) denotes the cyclic algebra with basis L v^i, v ell v^{-1}=sigma(ell), v^n=b. The group <sigma> acts simply transitively on Hom_k(L, k_s); for gamma in G_k and phi in Hom_k(L,k_s) there is a unique j in Z/n with gamma\u2218phi = phi\u2218sigma^j, independent of phi, and chi_L(gamma)=j/n defines a character chi_L in H^1(G_k, Q/Z). If L is a field, chi_L is the character with kernel G_L that takes value 1/n on elements restricting to sigma. (i) (L,sigma,b) is a central simple k-algebra of degree n and a cyclic crossed product. Conversely, every cyclic crossed product of degree n is isomorphic to some (L,sigma,b). If L is a field, (L,sigma,b) is split by L, and b -> [(L,sigma,b)] induces an isomorphism k^\u00d7/N_{L/k}(L^\u00d7) \u2245 Br(L/k). (ii) [(L,sigma,b)] + [(L,sigma,b')] = [(L,sigma,bb')]. (iii) The class of (L,sigma,b) depends only on chi_L and b. Writing it as [chi_L,b), it is additive in the character. More precisely, let e be a primitive idempotent of L, M=Le, and d=[M:k]. Then sigma^{n/d} generates the stabilizer of e, the image of chi_L is (1/d)Z/Z, chi_L = chi_M when M is given the generator sigma^{n/d}|_M, and (L,sigma,b) \u2245 M_{n/d}((M,sigma^{n/d}|_M,b)). In particular, L is a field if and only if chi_L has order n. (iv) Suppose L is a field and m divides n. Then m[chi_L,b) = [m chi_L,b) is the class of (L_{n/m}, sigma|_{L_{n/m}}, b), where L_{n/m} is the subfield of L of degree n/m. In characteristic p, for w in W_2(k), a in k, and b in k^\u00d7: (v) p[w,b) = [w_0,b); (vi) [w,b) + [a,b) = [w+V(a),b).",
    "reference": "Theorem 21",
    "tags": [
      "cyclic algebras",
      "brauer group",
      "characters",
      "witt symbols"
    ]
  },
  {
    "id": "ref_23",
    "slogan": "Exponent of a cyclic algebra is the order of its parameter modulo norms, giving a division criterion.",
    "formal statement": "Let L/k be a cyclic field extension of degree n with Galois group generated by sigma, and let (L,sigma,b) be the cyclic algebra with v ell v^{-1}=sigma(ell), v^n=b. Then exp((L,sigma,b)) equals the order of b in k^\u00d7/N_{L/k}(L^\u00d7). In particular, if this order is n, then (L,sigma,b) is a division algebra of exponent n.",
    "reference": "Theorem 22",
    "tags": [
      "cyclic algebras",
      "exponent",
      "division algebras",
      "norms"
    ]
  },
  {
    "id": "ref_24",
    "slogan": "Witt symbols over Laurent series fields give division algebras of degree p^2 and exponent p^2.",
    "formal statement": "Let F be a field of characteristic p and k=F((t)). Let w=(w_0,w_1) in W_2(F) with w_0 notin wp(F). Then [w,t) over k, where [w,t) denotes the cyclic algebra associated to the Witt vector w and parameter t, is a division algebra of degree p^2 and exponent p^2. Similarly, for a in F notin wp(F), [a,t) is a division algebra of degree p. Such w exist whenever wp(F) != F; for example, w=(w_0,0) with w_0 notin wp(F).",
    "reference": "Theorem 23",
    "tags": [
      "laurent series fields",
      "division algebras",
      "witt symbols",
      "exponent"
    ]
  },
  {
    "id": "ref_25",
    "slogan": "Index bound for tensor products when a common subfield splits one factor and embeds in the other.",
    "formal statement": "Let D be a central division k-algebra of degree d, let B be a central simple k-algebra, and let K/k be a field extension of degree m with m dividing d. If K splits B and K embeds in D, then ind(D tensor_k B) divides d.",
    "reference": "Theorem 24",
    "tags": [
      "index bounds",
      "tensor products",
      "division algebras",
      "splitting fields"
    ]
  },
  {
    "id": "ref_26",
    "slogan": "Construction of degree p^2 exponent p^2 division algebras by tensoring with a symbol algebra.",
    "formal statement": "Let char(k)=p. Let D be a division algebra of degree p^2 and exponent p^2 over k, and let B=[a,c) with a in k and c in k^\u00d7. Suppose that either k(wp^{-1}a) embeds in D with a notin wp(k), where k(wp^{-1}a) is the Artin-Schreier extension generated by a root of t^p-t=a, or k(c^{1/p}) embeds in D with c notin k^p. Then D tensor_k B is isomorphic to M_p(E) for a division algebra E of degree p^2 and exponent p^2, and [E]=[D]+[a,c) in Br(k). E is cyclic, hence not a candidate counterexample, in each of the following cases: (i) D=(L,sigma,b) is cyclic and k(wp^{-1}a) is contained in L; (ii) D=[w,b) and c=b.",
    "reference": "Theorem 25",
    "tags": [
      "division algebras",
      "tensor products",
      "symbol algebras",
      "exponent p^2"
    ]
  },
  {
    "id": "ref_27",
    "slogan": "For degree p^2 exponent p division algebras, symbol length two is equivalent to decomposability into two cyclic degree p factors.",
    "formal statement": "Let char(k)=p and let D be a division algebra of degree p^2 and exponent p. Define the symbol length lambda(D) as the least r such that [D]=sum_{i=1}^r [a_i,b_i) in Br(k); it is finite. Then lambda(D) >= 2, and D is isomorphic to a tensor product of two cyclic algebras of degree p if and only if lambda(D)=2. In that case D is a crossed product with group Z/p x Z/p. Consequently, a division algebra of degree p^2 and exponent p that is not a crossed product must satisfy lambda(D) >= 3. A tensor decomposition into two degree-p factors that are not known to be cyclic does not by itself give lambda(D)=2 when p>=5.",
    "reference": "Theorem 26",
    "tags": [
      "symbol length",
      "decomposability",
      "crossed products",
      "exponent p"
    ]
  },
  {
    "id": "ref_28",
    "slogan": "Generic division algebras from generic matrices and Amitsur specialization for crossed products.",
    "formal statement": "Let k_0 be a field and n>=2. Let X_1,...,X_r (r>=2) be n x n generic matrices whose entries are independent indeterminates over k_0. The k_0-algebra they generate is a domain, and its ring of central quotients UD(k_0,n) is a central division algebra of degree n over its center. If char(k_0)=0 and UD(k_0,n) is a G-crossed product for a group G of order n, then every central division algebra of degree n whose center contains k_0 is a G-crossed product. If the corresponding specialization statement holds for k_0=F_p and n=p^2, then the assertion that every central division p-algebra of degree p^2 is a crossed product is equivalent to UD(F_p,p^2) being a crossed product.",
    "reference": "Theorem 27",
    "tags": [
      "generic division algebras",
      "specialization",
      "crossed products",
      "central division algebras"
    ]
  },
  {
    "id": "ref_29",
    "slogan": "Saltman existence of noncrossed product division p-algebras in degree p^n for n>=3.",
    "formal statement": "For every prime p and every n>=3 there exist division p-algebras of degree p^n that are not crossed products.",
    "reference": "Theorem 28",
    "tags": [
      "noncrossed products",
      "p-algebras",
      "division algebras"
    ]
  },
  {
    "id": "ref_30",
    "slogan": "Albert degree p cyclicity criterion: degree p division algebras are cyclic exactly when they contain a p-central element, known for p=2,3.",
    "formal statement": "Let char(k)=p and let D be a central division k-algebra of degree p. Then D is cyclic if and only if D contains a p-central element, meaning an element x notin k with x^p in k. Every such D is known to be cyclic when p=2 or p=3.",
    "reference": "Theorem 29",
    "tags": [
      "cyclicity criteria",
      "p-central elements",
      "division algebras"
    ]
  },
  {
    "id": "ref_31",
    "slogan": "Amitsur-Saltman existence of division p-algebras without p-central elements in every degree p^n for n>=2.",
    "formal statement": "For every prime p and every n>=2 there exist division p-algebras of degree p^n that contain no p-central element (no element x notin k with x^p in k). They are abelian crossed products and they are not cyclic.",
    "reference": "Theorem 30",
    "tags": [
      "p-central elements",
      "p-algebras",
      "noncyclic division algebras"
    ]
  },
  {
    "id": "ref_32",
    "slogan": "Saltman normal maximal subfield lemma: normal maximal subfields imply Galois maximal subfields, and purely inseparable maximal subfields imply crossed products.",
    "formal statement": "Let D be a central division k-algebra. If D contains a maximal subfield that is normal over k, then D contains a maximal subfield that is Galois over k. In particular, if char(k)=p and D contains a purely inseparable maximal subfield, then D is a crossed product.",
    "reference": "Theorem 31",
    "tags": [
      "normal extensions",
      "galois maximal subfields",
      "crossed products"
    ]
  },
  {
    "id": "ref_33",
    "slogan": "Degree p^2 division algebras with a p-central element are crossed products when the centralizer over k(x) is cyclic.",
    "formal statement": "Let char(k)=p and let D be a central division k-algebra of degree p^2. Suppose D contains a p-central element x, and put K=k(x). If the division algebra C_D(K), which is central of degree p over K, is cyclic over K, then D is a crossed product. Consequently, a division algebra of degree p^2 that is not a crossed product either contains no p-central element, or has the property that C_D(k(x)) is a noncyclic division algebra of degree p over k(x) for every p-central x. When p=2 or p=3, the second alternative is impossible, so a counterexample for p=2 or p=3 contains no p-central element.",
    "reference": "Theorem 32",
    "tags": [
      "p-central elements",
      "centralizers",
      "crossed products",
      "degree p^2"
    ]
  },
  {
    "id": "ref_34",
    "slogan": "Non-division degree p^2 crossed product criterion in terms of the underlying degree p division algebra.",
    "formal statement": "Let char(k)=p and let A=M_p(D) with D a central division k-algebra of degree p. For a group G of order p^2, A is a G-crossed product if and only if D is cyclic or D is split by a Galois field extension of k with group G. Hence A is a crossed product if and only if D is cyclic or D is split by a Galois field extension of degree p^2. If D is cyclic, A is a G-crossed product for both groups of order p^2, and A contains a cyclic maximal subfield of degree p^2. In particular, A is always a crossed product when p=2 or p=3.",
    "reference": "Theorem 33",
    "tags": [
      "non-division algebras",
      "crossed products",
      "degree p^2",
      "galois splitting fields"
    ]
  },
  {
    "id": "ref_35",
    "slogan": "Rowen-Saltman prime-to-p extension results for crossed products in degree p^2 and higher.",
    "formal statement": "Every division algebra of degree p^2 becomes a crossed product after some scalar extension of degree prime to p. For degree p^3 and above there are noncrossed products that remain noncrossed products after every prime-to-p extension.",
    "reference": "Theorem 34",
    "tags": [
      "prime-to-p extensions",
      "crossed products",
      "division algebras"
    ]
  },
  {
    "id": "ref_36",
    "slogan": "Albert theorem for degree 4: all division algebras of degree 4 are crossed products, and characteristic 2 exponent 2 algebras are cyclic.",
    "formal statement": "Every central division algebra of degree 4 is a crossed product. In characteristic 2, every central simple algebra of degree 4 and exponent 2 is cyclic.",
    "reference": "Theorem 35",
    "tags": [
      "degree 4 algebras",
      "crossed products",
      "cyclic algebras"
    ]
  },
  {
    "id": "ref_37",
    "slogan": "Saltman theorem: cyclic division p-algebras are crossed products for every group of the same order.",
    "formal statement": "Let char(k)=p and let D be a cyclic central division k-algebra of degree p^d. Then D is a G-crossed product for every group G of order p^d. Consequently, for every r>=1, M_r(D) is a G-crossed product for every group G of order r p^d. In particular, a cyclic division p-algebra of degree p^2 is also a Z/p x Z/p crossed product.",
    "reference": "Theorem 36",
    "tags": [
      "cyclic p-algebras",
      "crossed products",
      "matrix algebras"
    ]
  },
  {
    "id": "ref_38",
    "slogan": "Hanke valued-field criterion linking inertial Galois maximal subfields, crossed products, and residue Galois maximal subfields.",
    "formal statement": "Let F be a valued field and let D be a finite-dimensional central division F-algebra such that the valuation of F extends to D. Suppose D is inertially split, that is, D tensor_F F^h has a splitting field inertial over the Henselization F^h. Consider: (1) D contains a maximal subfield that is inertial and Galois over F; (2) D is a crossed product; (3) the residue division algebra Dbar contains a maximal subfield that is Galois over the residue field Fbar. Then (1) implies (2) implies (3). If F is Henselian, (1), (2), and (3) are equivalent, and a Galois group G in (3) can be realized as the Galois group in (1). The residue field Fbar may be imperfect.",
    "reference": "Theorem 37",
    "tags": [
      "valued division algebras",
      "inertial splitting",
      "crossed products",
      "residue algebras"
    ]
  },
  {
    "id": "ref_39",
    "slogan": "McKinnie theorem: Galois subfields of nondegenerate semiramified p-algebras are inertial, with quotient restriction for generic abelian crossed products.",
    "formal statement": "Let F be a Henselian valued field of characteristic p and let D be a semiramified p-algebra over F with separable residue field that is not strongly degenerate in the sense of McKinnie's Definition 0.1.1. Then every Galois subfield of D is inertial over F. For the generic abelian crossed products of Amitsur-Saltman defined by a group G and a non-degenerate matrix, every Galois subfield has Galois group a quotient of G.",
    "reference": "Theorem 38",
    "tags": [
      "semiramified p-algebras",
      "galois subfields",
      "valued fields"
    ]
  },
  {
    "id": "ref_40",
    "slogan": "Tensor products of cyclic p-algebras are cyclic, and Witt symbols satisfy the standard additivity and norm-vanishing identities.",
    "formal statement": "Let char(k)=p. A tensor product of cyclic p-algebras is cyclic. Let m>=1, omega,omega' in W_m(k), a=(a_1,...,a_m) in W_m(k), and b,b' in k^\u00d7. With [omega,b)=(L_omega,sigma,b) as in the Witt-symbol notation, the following hold in Br(k): (i) [omega,b) + [omega',b) = [omega+omega',b); (ii) [omega,b) + [omega,b') = [omega,bb'); (iii) [(0,a_1,...,a_m),b) = [(a_1,...,a_m),b), where the left side is formed in W_{m+1}(k); (iv) [(b,0,...,0),b) = 0; (v) [omega,b)=0 if b in (k^\u00d7)^{p^m}; if omega_0 notin wp(k), then [omega,b)=0 if and only if b is a norm from the field L_omega=k(wp^{-1}omega).",
    "reference": "Theorem 39",
    "tags": [
      "witt symbols",
      "tensor products",
      "brauer group",
      "cyclic p-algebras"
    ]
  },
  {
    "id": "ref_41",
    "slogan": "Indecomposable crossed products of degree p^2 and exponent p exist for odd p, while p=2 has no such examples.",
    "formal statement": "Let p be an odd prime. There exist division p-algebras of degree p^2 and exponent p that are indecomposable and are nevertheless crossed products (generic abelian crossed products). Indecomposability therefore does not obstruct being a crossed product. For p=2 no such algebras exist: by Albert's theorem, every central simple algebra of degree 4 and exponent 2 is a tensor product of two quaternion algebras.",
    "reference": "Theorem 40",
    "tags": [
      "indecomposable division algebras",
      "crossed products",
      "exponent p"
    ]
  },
  {
    "id": "ref_42",
    "slogan": "Exact sequence relating Kato-Milne cohomology groups of consecutive p-power exponent via the Shift and Exp maps.",
    "formal statement": "Let $F$ be a field of characteristic $p$ and let $m, n$ be positive integers. Then the sequence $$0 \\to \\operatorname{H}^{n+1}_{p^{m-1}}(F) \\xrightarrow{\\operatorname{Shift}} \\operatorname{H}^{n+1}_{p^m}(F) \\xrightarrow{\\operatorname{Exp}} \\operatorname{H}^{n+1}_{p}(F) \\to 0$$ is exact, where Shift is the natural inclusion and Exp is the map sending a symbol to its $p^{m-1}$-fold sum.",
    "reference": "Section 2 (Preliminaries), Theorem (Exact)",
    "tags": [
      "kato-milne cohomology",
      "exact sequence",
      "witt vectors"
    ]
  },
  {
    "id": "ref_43",
    "slogan": "Existence of indecomposable p-algebras of large symbol length that remain indecomposable under any prime-to-p extension.",
    "formal statement": "Let $p$ be a prime, $k$ a field of characteristic $p$, and $m, \\ell$ positive integers with $\\ell \\geq 2$, excluding the case $p = \\ell = 2$ and $m = 1$. Then there exists a field $F$ containing $k$ and a $p$-algebra $A$ of degree $p^{\\ell m}$ and exponent $p^m$ over $F$ such that $\\operatorname{sl}_{p^m}([A_L]) \\geq \\ell + 1$ for all prime-to-$p$ field extensions $L$ of $F$.",
    "reference": "Section 3, Proposition (indecomposable)",
    "tags": [
      "central simple algebras",
      "symbol length",
      "indecomposable algebras"
    ]
  },
  {
    "id": "ref_44",
    "slogan": "The p-rank of a field of characteristic p bounds the symbol length of every p^m-torsion Brauer class.",
    "formal statement": "Let $p$ be a prime, $F$ a field of characteristic $p$ with $\\operatorname{rank}_p(F) = r < \\infty$, and $A$ a $p$-algebra of exponent $p^m$ over $F$. Then $\\operatorname{sl}_{p^m}([A]) \\leq r$.",
    "reference": "Section 3, Proposition (prank)",
    "tags": [
      "brauer group",
      "symbol length",
      "p-rank"
    ]
  },
  {
    "id": "ref_45",
    "slogan": "Every class in Kato-Milne cohomology over a field of finite p-rank is a sum of at most binomial(r,n) symbols.",
    "formal statement": "Let $F$ be a field of characteristic $p$ with finite $p$-rank $r$. Then the symbol length of any class in $\\operatorname{H}^{n+1}_{p^m}(F)$ is at most $\\binom{r}{n}$, and in particular the symbol length of any class in $\\operatorname{Br}_{p^m}(F)$ is at most $r$.",
    "reference": "Section 3, Corollary (after the p-basis expansion lemma)",
    "tags": [
      "kato-milne cohomology",
      "symbol length",
      "p-rank"
    ]
  },
  {
    "id": "ref_46",
    "slogan": "The symbol length of Kato-Milne cohomology of exponent p^{m+1} is bounded by the sum of the symbol lengths at exponents p^m and p.",
    "formal statement": "Let $F$ be a field of characteristic $p$, and suppose the symbol length of $\\operatorname{H}^{n+1}_{p^m}(F)$ is $t$ and the symbol length of $\\operatorname{H}^{n+1}_{p}(F)$ is $s$, both finite. Then the symbol length of $\\operatorname{H}^{n+1}_{p^{m+1}}(F)$ is at most $t + s$.",
    "reference": "Section 4, Proposition (first proposition)",
    "tags": [
      "kato-milne cohomology",
      "symbol length",
      "upper bounds"
    ]
  },
  {
    "id": "ref_47",
    "slogan": "The symbol length of Kato-Milne cohomology of exponent p^m is bounded by m times the symbol length at exponent p.",
    "formal statement": "Let $F$ be a field of characteristic $p$, and suppose the symbol length of $\\operatorname{H}^{n+1}_{p}(F)$ is $s < \\infty$. Then the symbol length of $\\operatorname{H}^{n+1}_{p^m}(F)$ is at most $m \\cdot s$.",
    "reference": "Section 4, Corollary (after the t+s proposition)",
    "tags": [
      "kato-milne cohomology",
      "symbol length",
      "upper bounds"
    ]
  },
  {
    "id": "ref_48",
    "slogan": "The p-rank of a finitely generated extension of a field of characteristic p is the sum of the base p-rank and the transcendence degree.",
    "formal statement": "Let $k$ be a field of positive characteristic $p$ with $\\operatorname{rank}_p(k) = r$, and let $F$ be a finitely generated field extension of $k$ of transcendence degree $t$. Then $\\operatorname{rank}_p(F) = r + t$.",
    "reference": "Section 5, Lemma (Jarden)",
    "tags": [
      "p-rank",
      "field extensions",
      "transcendence degree"
    ]
  },
  {
    "id": "ref_49",
    "slogan": "The essential dimension of a Brauer class plus the p-rank of the base field is at least the symbol length of the class.",
    "formal statement": "Let $k$ be a field of characteristic $p > 0$ with $\\operatorname{rank}_p(k) = r$, let $F$ be a field containing $k$, and let $A$ be a $p$-algebra of exponent $p^m$ ($m \\geq 1$) over $F$. Then $\\operatorname{ed}_{\\operatorname{Br}_{p^m}}([A]) + r \\geq \\operatorname{sl}_{p^m}([A])$.",
    "reference": "Section 5, Lemma (symbol)",
    "tags": [
      "essential dimension",
      "symbol length",
      "brauer group"
    ]
  },
  {
    "id": "ref_50",
    "slogan": "The essential p-dimension of the functor of p-algebras of degree p^{\u2113m} and exponent p^m is at least \u2113+1 when the base field is perfect.",
    "formal statement": "Let $k$ be a field of characteristic $p > 0$ with $\\operatorname{rank}_p(k) = r$, and let $m, \\ell$ be positive integers with $\\ell \\geq 2$. Then $\\operatorname{ed}(\\operatorname{Alg}_{p^{\\ell m},\\, p^m};\\, p) \\geq \\ell + 1 - r$. In particular, when $k$ is perfect, $\\operatorname{ed}(\\operatorname{Alg}_{p^{\\ell m},\\, p^m};\\, p) \\geq \\ell + 1$.",
    "reference": "Section 5, Theorem (Alg)",
    "tags": [
      "essential dimension",
      "central simple algebras",
      "p-algebras",
      "lower bounds"
    ]
  },
  {
    "id": "ref_51",
    "slogan": "The generic sum of \u2113 symbols in Kato-Milne cohomology over an algebraically closed field of characteristic p has essential p-dimension at least \u2113+n.",
    "formal statement": "Let $p$ be a prime, $k$ an algebraically closed field of characteristic $p$, and $m, n, \\ell \\geq 1$ integers. Let $F_{\\ell,m,n} = k(x_{1,1}, \\ldots, y_{\\ell,n})$ be the rational function field in $(m+n)\\ell$ indeterminates over $k$, and let $A_{\\ell,m,n} = \\sum_{i=1}^{\\ell} (x_{i,1}, \\ldots, x_{i,m}) \\otimes y_{i,1} \\otimes \\cdots \\otimes y_{i,n} \\in \\operatorname{H}^{n+1}_{p^m}(F_{\\ell,m,n})$. Then $\\operatorname{ed}_{\\operatorname{H}^{n+1}_{p^m}}(A_{\\ell,m,n};\\, p) \\geq \\ell + n$.",
    "reference": "Section 6, Theorem (GenSum)",
    "tags": [
      "essential dimension",
      "kato-milne cohomology",
      "generic symbols",
      "lower bounds"
    ]
  },
  {
    "id": "ref_52",
    "slogan": "The essential dimension of a p-group of exponent dividing p^m minimally generated by \u2113 elements is at most m, when the base field is large enough.",
    "formal statement": "Let $m, \\ell$ be positive integers, $k$ a field of characteristic $p$ with $|k| \\geq p^\\ell$, and $G$ a $p$-group of exponent dividing $p^m$ minimally generated by $\\ell$ elements. Then $\\operatorname{ed}(G) \\leq m$.",
    "reference": "Section 6, Lemma (Ledet)",
    "tags": [
      "essential dimension",
      "p-groups",
      "galois extensions"
    ]
  },
  {
    "id": "ref_53",
    "slogan": "A sum of \u2113 symbols in Kato-Milne cohomology over a sufficiently large base field has essential dimension at most m + \u2113n.",
    "formal statement": "Let $k$ be a field of characteristic $p$ with $|k| \\geq p^\\ell$, $F$ a field containing $k$, and $\\pi$ a sum of $\\ell$ symbols in $\\operatorname{H}^{n+1}_{p^m}(F)$. Then $\\operatorname{ed}_{\\operatorname{H}^{n+1}_{p^m}}(\\pi) \\leq m + \\ell n$.",
    "reference": "Section 6, Proposition (upper bound)",
    "tags": [
      "essential dimension",
      "kato-milne cohomology",
      "upper bounds"
    ]
  },
  {
    "id": "ref_54",
    "slogan": "A sum of \u2113 symbols in the p^m-torsion Brauer group over a sufficiently large base field has essential dimension at most m + \u2113.",
    "formal statement": "Let $k$ be a field of characteristic $p$ with $|k| \\geq p^\\ell$, $F$ a field containing $k$, and $\\pi$ a sum of $\\ell$ symbols in $\\operatorname{Br}_{p^m}(F)$. Then $\\operatorname{ed}_{\\operatorname{Br}_{p^m}}(\\pi) \\leq m + \\ell$.",
    "reference": "Section 6, Corollary (after Proposition (upper bound))",
    "tags": [
      "essential dimension",
      "brauer group",
      "upper bounds"
    ]
  },
  {
    "id": "ref_55",
    "slogan": "Over an infinite perfect base field, the essential dimension of a Brauer class is sandwiched between its symbol length and its symbol length plus m.",
    "formal statement": "Let $k$ be an infinite perfect field of characteristic $p$, $F$ a field containing $k$, $m$ a positive integer, and $[A]$ a class in $\\operatorname{Br}_{p^m}(F)$. Then $$\\operatorname{sl}_{p^m}([A]) \\leq \\operatorname{ed}_{\\operatorname{Br}_{p^m}}([A]) \\leq \\operatorname{sl}_{p^m}([A]) + m.$$",
    "reference": "Section 6, Corollary (Bounds)",
    "tags": [
      "essential dimension",
      "symbol length",
      "brauer group",
      "upper bounds",
      "lower bounds"
    ]
  },
  {
    "id": "ref_56",
    "slogan": "The essential dimension of a p-algebra of degree p^n and exponent p^m is at most p^n + m \u2212 1; in particular, degree-8 exponent-2 algebras have essential dimension at most 5.",
    "formal statement": "Let $k$ be an infinite perfect field of characteristic $p$, $F$ a field containing $k$, and $A$ a $p$-algebra of degree $p^n$ and exponent $p^m$ over $F$. Then $\\operatorname{ed}_{\\operatorname{Br}_{p^m}}([A]) \\leq p^n + m - 1$. In particular, if $p = 2$ and $A$ has degree $8$ and exponent $2$, then $\\operatorname{ed}_{\\operatorname{Br}_{2}}([A]) \\leq 5$.",
    "reference": "Section 6, Corollary (final, after Corollary (Bounds))",
    "tags": [
      "essential dimension",
      "central simple algebras",
      "upper bounds"
    ]
  },
  {
    "id": "ref_57",
    "slogan": "In characteristic 2, the essential dimension and essential 2-dimension of degree-4 exponent-2 central simple algebras are both 3.",
    "formal statement": "Let k be an algebraically closed field of characteristic 2, and let Alg_{4,2} be the functor sending a field F containing k to the isomorphism classes of central simple F-algebras of degree 4 and exponent dividing 2. Then ed(Alg_{4,2}) = ed_2(Alg_{4,2}) = 3.",
    "reference": "Section 3, Proposition Alg42",
    "tags": [
      "essential dimension",
      "central simple algebras",
      "characteristic 2"
    ]
  },
  {
    "id": "ref_58",
    "slogan": "In characteristic 2, the essential dimension and essential 2-dimension of degree-4 exponent-4 central simple algebras lie between 4 and 5.",
    "formal statement": "Let k be an algebraically closed field of characteristic 2, and let Alg_{4,4} be the functor sending a field F containing k to the isomorphism classes of central simple F-algebras of degree 4 and exponent dividing 4. Then 4 \u2264 ed(Alg_{4,4}) \u2264 5 and 4 \u2264 ed_2(Alg_{4,4}) \u2264 5.",
    "reference": "Section 3, second proposition",
    "tags": [
      "essential dimension",
      "central simple algebras",
      "characteristic 2"
    ]
  },
  {
    "id": "ref_59",
    "slogan": "A quaternion algebra over a separable quadratic extension of a characteristic-2 field with trivial corestriction descends to the base field.",
    "formal statement": "Let F be a field of characteristic 2, let K = F[i : i^2 + i = \u03b1] be a separable quadratic extension of F, and let Q = [a, bi + c)_{2,K} be a quaternion algebra over K with a,b,c \u2208 F. If cor_{K/F}(Q) is trivial in Br(F), then Q descends to F; that is, there exists a quaternion F-algebra Q_0 such that Q_0 \u2297_F K \u2245 Q.",
    "reference": "Section 4, Lemma Baeklike",
    "tags": [
      "quaternion algebras",
      "corestriction",
      "characteristic 2"
    ]
  },
  {
    "id": "ref_60",
    "slogan": "Every central simple algebra of degree 8 and exponent 2 over a characteristic-2 field has essential dimension at most 8.",
    "formal statement": "Let k be an algebraically closed field of characteristic 2, let F be a field containing k, and let A be a central simple F-algebra of degree 8 and exponent 2. Then ed(A) \u2264 8.",
    "reference": "Section 4, Theorem EDD8E2",
    "tags": [
      "essential dimension",
      "central simple algebras",
      "characteristic 2"
    ]
  },
  {
    "id": "ref_61",
    "slogan": "The essential dimension of central simple algebras of degree p^{mn} and exponent p^m is at least n+1.",
    "formal statement": "Let k be an algebraically closed field of characteristic p>0, and let Alg_{p^{mn},p^m} be the functor sending a field F containing k to the isomorphism classes of central simple F-algebras of degree p^{mn} and exponent dividing p^m. Then ed(Alg_{p^{mn},p^m}) \u2265 n+1.",
    "reference": "Section 5, paragraph before the definition of LCA_{p^m,n}",
    "tags": [
      "essential dimension",
      "central simple algebras",
      "lower bounds"
    ]
  },
  {
    "id": "ref_62",
    "slogan": "Linked pairs of cyclic p-algebras have essential dimension 3.",
    "formal statement": "Let k be an algebraically closed field of characteristic p>0, and let LCA_{p,2} be the functor sending a field F containing k to pairs of linked cyclic F-algebras of degree p, i.e. pairs ([\u03b1,\u03b2_1)_{p,F}, [\u03b1,\u03b2_2)_{p,F}) with common \u03b1 \u2208 F and \u03b2_1,\u03b2_2 \u2208 F^\u00d7. Then ed(LCA_{p,2}) = 3.",
    "reference": "Section 5, paragraph after the discussion of LCA_{2,2}",
    "tags": [
      "essential dimension",
      "cyclic algebras",
      "linked algebras"
    ]
  },
  {
    "id": "ref_63",
    "slogan": "Inseparably linked sequences of cyclic p^m-algebras have essential dimension at most m+1 and essential p-dimension at most 2.",
    "formal statement": "Let k be an algebraically closed field of characteristic p>0. Let LCA_{p^m,n} be the functor sending a field F containing k to n-tuples (C_1,...,C_n) of linked cyclic F-algebras of degree p^m, i.e. C_i = [\u03c9, \u03b2_i)_{p^m,F} for a common \u03c9 \u2208 W_mF and \u03b2_i \u2208 F^\u00d7. If (C_1,...,C_n) is inseparably linked, meaning C_i = [\u03c9_i, \u03b2)_{p^m,F} for a fixed \u03b2 \u2208 F^\u00d7, then ed(C_1,...,C_n) \u2264 m+1 and ed_p(C_1,...,C_n) \u2264 2.",
    "reference": "Section 5, corollary after the definition of inseparably linked sequences",
    "tags": [
      "essential dimension",
      "cyclic algebras",
      "linked algebras"
    ]
  },
  {
    "id": "ref_64",
    "slogan": "Linked sequences of n quaternion algebras in characteristic 2 have essential dimension n+1.",
    "formal statement": "Let k be an algebraically closed field of characteristic 2, and let LCA_{2,n} be the functor sending a field F containing k to n-tuples of linked quaternion F-algebras, i.e. tuples ([\u03b1,\u03b2_1)_{2,F},...,[\u03b1,\u03b2_n)_{2,F}) with common \u03b1 \u2208 F and \u03b2_i \u2208 F^\u00d7. Then ed(LCA_{2,n}) = ed_2(LCA_{2,n}) = n+1.",
    "reference": "Section 5, Proposition seqquat",
    "tags": [
      "essential dimension",
      "quaternion algebras",
      "linked algebras"
    ]
  },
  {
    "id": "ref_65",
    "slogan": "The essential p-dimension of decomposable algebras that are tensor products of n cyclic p^m-algebras is n+1.",
    "formal statement": "Let k be an algebraically closed field of characteristic p>0, and let Dec_{p^m,n} be the functor sending a field F containing k to the isomorphism classes of central simple F-algebras of degree p^{mn} that are tensor products of n cyclic F-algebras of degree p^m. Then ed_p(Dec_{p^m,n}) = n+1.",
    "reference": "Section 6, theorem",
    "tags": [
      "essential p-dimension",
      "decomposable algebras",
      "cyclic algebras"
    ]
  },
  {
    "id": "ref_66",
    "slogan": "The essential p-dimension of cyclic extensions of degree p^m is 1.",
    "formal statement": "Let k be an algebraically closed field of characteristic p>0, and let H^1_{p^m} be the functor sending a field F containing k to the set of isomorphism classes of cyclic extensions of F of degree p^m, equivalently the Izhboldin group W_mF / wp(W_mF). Then ed_p(H^1_{p^m}) = 1.",
    "reference": "Section 7, paragraph before Proposition p2",
    "tags": [
      "essential p-dimension",
      "cyclic extensions",
      "Izhboldin groups"
    ]
  },
  {
    "id": "ref_67",
    "slogan": "Every class in H^1_{p^2} becomes of the form (c,0) after a prime-to-p extension of degree at most p^2-p+1.",
    "formal statement": "Let F be a field of characteristic p>0 and let (a,b) \u2208 H^1_{p^2}(F), the Izhboldin group describing cyclic extensions of degree p^2. Then there exists a field extension L/F of degree prime to p with [L:F] \u2264 p^2-p+1 and an element c \u2208 L such that the image of (a,b) in H^1_{p^2}(L) equals (c,0).",
    "reference": "Section 7, Proposition p2",
    "tags": [
      "Izhboldin groups",
      "cyclic extensions",
      "prime-to-p extensions"
    ]
  },
  {
    "id": "ref_68",
    "slogan": "Over a p-special field, every central simple algebra of prime degree p is cyclic.",
    "formal statement": "Let p be a prime and let F be a p-special field, i.e. a field with no finite extension of degree prime to p. Then every central simple F-algebra of degree p is cyclic.",
    "reference": "Section 8, theorem citing Albert",
    "tags": [
      "p-special fields",
      "cyclic algebras",
      "central simple algebras"
    ]
  },
  {
    "id": "ref_69",
    "slogan": "The essential p-dimension of prime-degree central simple algebras in bad characteristic is 2.",
    "formal statement": "Let k be an algebraically closed field of characteristic p>0, and let Alg_{p,p} be the functor sending a field F containing k to the isomorphism classes of central simple F-algebras of degree p and exponent dividing p. Then ed_p(Alg_{p,p}) = 2.",
    "reference": "Section 8, Corollary Algpp",
    "tags": [
      "essential p-dimension",
      "central simple algebras",
      "prime degree"
    ]
  },
  {
    "id": "ref_70",
    "slogan": "In characteristic p, the p-torsion of the Brauer group is generated by cyclic algebras of degree p.",
    "formal statement": "Let F be a field of characteristic p>0. Then the p-torsion subgroup pBr(F) of Br(F) is generated by the Brauer classes of cyclic F-algebras of degree p.",
    "reference": "Section 9, opening paragraph",
    "tags": [
      "Brauer group",
      "cyclic algebras",
      "characteristic p"
    ]
  },
  {
    "id": "ref_71",
    "slogan": "Decomposable algebras that are tensor products of n cyclic degree-p algebras have essential dimension at most n+1.",
    "formal statement": "Let k be an algebraically closed field of characteristic p>0, and let Dec_{p,n} be the functor sending a field F containing k to the isomorphism classes of central simple F-algebras of degree p^n that are tensor products of n cyclic F-algebras of degree p. Then ed(Dec_{p,n}) \u2264 n+1.",
    "reference": "Section 9, proposition before the lower bound for Alg_{p^n,p}",
    "tags": [
      "essential dimension",
      "decomposable algebras",
      "cyclic algebras"
    ]
  },
  {
    "id": "ref_72",
    "slogan": "The essential p-dimension of central simple algebras of degree p^n and exponent p is at least n+1.",
    "formal statement": "Let k be an algebraically closed field of characteristic p>0, and let Alg_{p^n,p} be the functor sending a field F containing k to the isomorphism classes of central simple F-algebras of degree p^n and exponent dividing p. Then ed_p(Alg_{p^n,p}) \u2265 n+1.",
    "reference": "Section 9, Proposition Algpnp",
    "tags": [
      "essential p-dimension",
      "central simple algebras",
      "lower bounds"
    ]
  },
  {
    "id": "ref_73",
    "slogan": "The projective stabilizer of the nine inflection points of a smooth plane cubic is the semidirect product of its 3-torsion translation group with SL_2(F_3).",
    "formal statement": "Let Fbar be an algebraically closed field of characteristic not 2 or 3, let Cbar be a smooth cubic curve in P^2(Fbar), and let I be its nine inflection points. Let S be the subgroup of PGL_3(Fbar) acting on I by translation by the nine 3-torsion points of the Jacobian E(Cbar), equivalently the image of the Heisenberg group generated by tilde x, tilde y with tilde x^3 = tilde y^3 = 1 and tilde x tilde y = rho tilde y tilde x. Then the stabilizer of I in PGL_3(Fbar) is the semidirect product S rtimes SL_2(F_3).",
    "reference": "Section Classical Invariant Theory, Theorem labelled classical",
    "tags": [
      "plane cubics",
      "inflection points",
      "projective groups"
    ]
  },
  {
    "id": "ref_74",
    "slogan": "The colinearity-preserving bijections of the nine inflection points form an affine group with kernel the 3-torsion group and quotient GL_2(F_3).",
    "formal statement": "Let Fbar be an algebraically closed field of characteristic not 2 or 3, let I be the set of nine inflection points of a smooth cubic curve in P^2(Fbar), let E(I) be the associated (Z/3Z)^2 group acting on I, and let Aff(I) be the group of bijections of I preserving the colinearity relation. Then there is an exact sequence 0 -> E(I) -> Aff(I) -> GL_2(F_3) -> 0, where the map Aff(I) -> GL_2(F_3) is the induced action on E(I).",
    "reference": "Section Classical Invariant Theory, Lemma 10",
    "tags": [
      "inflection points",
      "affine groups",
      "finite groups"
    ]
  },
  {
    "id": "ref_75",
    "slogan": "The affine group of the nine inflection points has a canonical cohomology class whose restriction to the 3-torsion kernel is the identity.",
    "formal statement": "Let Fbar be an algebraically closed field of characteristic not 2 or 3, let I be the set of nine inflection points of a smooth cubic curve in P^2(Fbar), let E(I) be the associated (Z/3Z)^2 group acting on I, let Aff(I) be the group of bijections of I preserving colinearity, and let SAff(I) be the preimage of SL_2(F_3) in Aff(I) under the natural map to GL_2(F_3). There is a unique class gamma in H^1(Aff(I), E(I)) whose restriction to E(I) is the identity element of H^1(E(I), E(I)) = Hom(E(I), E(I)). The same uniqueness holds for the restriction gamma_S in H^1(SAff(I), E(I)).",
    "reference": "Section Classical Invariant Theory, Proposition labelled gamma",
    "tags": [
      "Galois cohomology",
      "affine groups",
      "3-torsion"
    ]
  },
  {
    "id": "ref_76",
    "slogan": "Nine-point inflection configurations and their Heisenberg translation subgroups in PGL_3 are in one-to-one correspondence.",
    "formal statement": "Let Fbar be an algebraically closed field of characteristic not 2 or 3 and let rho be a primitive cube root of unity. Let Ical be the set of nine-point subsets I of P^2(Fbar) that occur as the inflection points of a smooth cubic curve. Let Scal be the set of nine-element subgroups S of PGL_3(Fbar) that are images of subgroups of GL_3(Fbar) generated by tilde x, tilde y with tilde x^3 = tilde y^3 = 1 and tilde x tilde y = rho tilde y tilde x. Then the assignment I -> S, where S is the translation subgroup of the Jacobian acting on I, gives a bijection between Ical and Scal.",
    "reference": "Section Classical Invariant Theory, Theorem labelled theom:9",
    "tags": [
      "inflection points",
      "Heisenberg groups",
      "projective transformations"
    ]
  },
  {
    "id": "ref_77",
    "slogan": "The stabilizer of the nine inflection points is exactly the normalizer of their translation subgroup in PGL_3.",
    "formal statement": "Let Fbar be an algebraically closed field of characteristic not 2 or 3, let I be a nine-point inflection set in P^2(Fbar), let S be its associated translation subgroup of PGL_3(Fbar), and let H be the stabilizer of I in PGL_3(Fbar). Then H is the normalizer of S in PGL_3(Fbar).",
    "reference": "Section Classical Invariant Theory, Theorem labelled theom:11",
    "tags": [
      "projective groups",
      "normalizers",
      "inflection points"
    ]
  },
  {
    "id": "ref_78",
    "slogan": "A plane cubic contains a given nine-point inflection set exactly when the corresponding Heisenberg subgroup fixes the cubic.",
    "formal statement": "Let Fbar be an algebraically closed field of characteristic not 2 or 3, let I be a nine-point inflection set in P^2(Fbar), let S be its associated translation subgroup of PGL_3(Fbar), and let tilde S be the Heisenberg preimage of S in GL_3(Fbar). For a cubic curve Cbar in P^2(Fbar), the following are equivalent: I is contained in Cbar; S fixes Cbar; tilde S fixes Cbar.",
    "reference": "Section Classical Invariant Theory, Theorem labelled theom:12",
    "tags": [
      "plane cubics",
      "Heisenberg groups",
      "fixed curves"
    ]
  },
  {
    "id": "ref_79",
    "slogan": "The identification of the translation subgroup with 3-torsion is compatible with the normalizer action and preserves the Weil pairing.",
    "formal statement": "Let Fbar be an algebraically closed field of characteristic not 2 or 3, let Cbar be a smooth cubic curve in P^2(Fbar) containing a nine-point inflection set I, let E = E(Cbar), let S be the associated translation subgroup of PGL_3(Fbar), let S^+ be the stabilizer of Cbar in PGL_3(Fbar), and let phi_C:S -> E[3] be the map s -> s(P)-P for P in I. Let psi:S -> E(I) be the natural isomorphism. If g in PGL_3(Fbar) normalizes S and bar g denotes its image in SL_2(F_3) acting on E(I) and S by conjugation, then: (a) bar g circ psi = psi circ bar g; (b) there is a choice of phi_{g(Cbar)} such that phi_{g(Cbar)} = phi_C circ bar g; (c) if Cbar and Cbar' have isomorphic Jacobians identified, and the corresponding maps phi_{Cbar'}, phi_{Cbar}: S -> E[3] differ by an automorphism of E, then Cbar' = Cbar as subvarieties of P^2; (d) phi_C preserves the Weil pairing.",
    "reference": "Section Classical Invariant Theory, Proposition labelled prop:14",
    "tags": [
      "3-torsion",
      "Weil pairing",
      "normalizer"
    ]
  },
  {
    "id": "ref_80",
    "slogan": "An elliptic curve is the Jacobian of a genus-one curve in a Severi-Brauer surface exactly when the algebra contains a Galois-stable skew-commuting 3-torsion subgroup matching the elliptic curve 3-torsion.",
    "formal statement": "Let F be a field of characteristic not 2 or 3, let E/F be an elliptic curve, let A/F be a degree-three central simple algebra, and let K/F be the Galois extension obtained by adjoining all points of E[3], with G = Gal(K/F). Then there exists a smooth genus-one curve C contained in SB(A) with E(C) = E if and only if there exists a subgroup S of (A tensor_F K)^* / K^* such that S is preserved by G, S is generated by the images of elements x', y' in A tensor_F K with x' y' = rho y' x' for a primitive cube root rho, and S is isomorphic to E[3] as a G-module, preserving the pairing (the pairing on S induced by the skew-commutation relation and the Weil pairing on E[3]).",
    "reference": "Section General F, Theorem labelled theom:16",
    "tags": [
      "Severi-Brauer surfaces",
      "elliptic curves",
      "3-torsion",
      "division algebras"
    ]
  },
  {
    "id": "ref_81",
    "slogan": "The pullback of the canonical affine cohomology class gives the principal homogeneous space class of the genus-one curve.",
    "formal statement": "Let F be a field of characteristic not 2 or 3, let Fbar be its separable closure, let Gbar = Gal(Fbar/F), let I be a nine-point inflection set in P^2(Fbar) defined over F, let Cbar be a smooth genus-one curve over Fbar containing I whose Jacobian E is defined over F, let Phi: Gbar -> Aff(I) be the homomorphism induced by the Galois action on I, and let gamma in H^1(Aff(I), E(I)) be the canonical class. Then Phi^*(gamma) in H^1(Gbar, E[3]) maps to the class gamma' in H^1(Gbar, E) that defines Cbar as a principal homogeneous space over E.",
    "reference": "Section General F, unnumbered Theorem after the discussion of the cocycle e(sigma)",
    "tags": [
      "Galois cohomology",
      "principal homogeneous spaces",
      "elliptic curves"
    ]
  },
  {
    "id": "ref_82",
    "slogan": "The same canonical cohomology class maps to the Brauer class of the Severi-Brauer surface.",
    "formal statement": "Let F be a field of characteristic not 2 or 3, let A/F be a degree-three central simple algebra, let Fbar be its separable closure, let Gbar = Gal(Fbar/F), let I be a nine-point inflection set in SB(A), let Phi: Gbar -> Aff(I) be the induced homomorphism, let gamma be the canonical class in H^1(Aff(I), E(I)), and let eta: H^1(Aff(I), E(I)) -> H^2(Gbar, Fbar^*) be the composition of Phi^* with the usual connecting map to the Brauer group. Then eta(gamma) in H^2(Gbar, Fbar^*) is the Brauer class of A/F. Consequently, if C is contained in SB(A), then C is defined by the image of a class in H^1(Gbar, E(C)[3]) that maps to the Brauer class of A/F.",
    "reference": "Section General F, unnumbered Proposition immediately after the theorem on Phi^*(gamma)",
    "tags": [
      "Brauer group",
      "Galois cohomology",
      "Severi-Brauer varieties"
    ]
  },
  {
    "id": "ref_83",
    "slogan": "If all 3-torsion of E is defined over F(rho), then E occurs as the Jacobian of a genus-one curve in every Severi-Brauer surface of a degree-three algebra.",
    "formal statement": "Let F be a field of characteristic not 2 or 3, let rho be a primitive cube root of unity, let E/F be an elliptic curve whose full 3-torsion E[3] is defined over F(rho), and let A/F be a degree-three central simple algebra. Then there exists a smooth genus-one curve C contained in SB(A) such that E(C) = E.",
    "reference": "Section Examples, Corollary labelled cor:18",
    "tags": [
      "elliptic curves",
      "Severi-Brauer surfaces",
      "3-torsion"
    ]
  },
  {
    "id": "ref_84",
    "slogan": "When the 3-torsion field is cyclic cubic, occurrence is equivalent to splitting the algebra by a cubic extension with trivial relative cyclic algebra.",
    "formal statement": "Let F be a field of characteristic not 2 or 3, let E/F be an elliptic curve, let K/F be the field obtained by adjoining all points of E[3], and assume K/F is cyclic of degree 3. Let A/F be a degree-three central simple algebra. Then there exists a smooth genus-one curve C contained in SB(A) with E(C) = E if and only if A is split by a field F(a^{1/3}) for some a in F^* such that the cyclic algebra (a, K/F) is trivial.",
    "reference": "Section Examples, Proposition labelled prop:17",
    "tags": [
      "cyclic algebras",
      "elliptic curves",
      "splitting fields"
    ]
  },
  {
    "id": "ref_85",
    "slogan": "When the 3-torsion field is quadratic and F contains rho, occurrence is equivalent to the existence of an S_3-Galois splitting field over F.",
    "formal statement": "Let F be a field of characteristic not 2 or 3 containing a primitive cube root rho, let E/F be an elliptic curve, let K/F be the field obtained by adjoining all points of E[3], and assume K/F is cyclic of order 2. Let D/F be a degree-three division algebra. Then there exists a smooth genus-one curve C contained in SB(D) with E(C) = E if and only if there exists a field L/K splitting D such that L/F is Galois with group S_3, the dihedral group of order 6.",
    "reference": "Section Examples, Corollary labelled cor:19",
    "tags": [
      "division algebras",
      "S_3 extensions",
      "elliptic curves"
    ]
  }
]
