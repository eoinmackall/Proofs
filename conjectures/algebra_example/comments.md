**What is true in the tame case (characteristic not dividing the degree).** Noncrossed product division algebras of degree p² exist for odd p: Amitsur in characteristic 0, Schacher–Small in characteristic ℓ ≠ p, and the explicit constructions of Hanke and Coyette. The obstruction is that a tamely and totally ramified cyclic extension of degree e forces μ_e into the residue field. Coyette starts with an algebra A of degree p² over a global field, containing a cyclic degree-p subfield K, and uses two places. At the first, K is totally ramified and the residue field contains no μ_{p²}, which excludes C_{p²}. At the second, K is inert and the residue field is finite of order ≢ 0, 1 (mod p), which excludes C_p × C_p. A twisted Laurent series construction then produces a noncrossed product. Its transfer lemma requires the characteristic not to divide the residue degree.

**What is available in our case (characteristic p, degree p², p > 2).**

* **Galois cohomology.** Since cd_p(k) ≤ 1, the σ-fixed classes in K/℘(K) are exactly res(k/℘k) + F_p·β. With `lemma_AS_galois_compatibility_field` and the argument forcing λ = 1, this gives a proposed criterion not yet in your file: D is a noncrossed product if and only if, for every cyclic degree-p subfield K′ ⊆ D, the class [D ⊗_k K′] is not [b, x)_{K′} for any nonzero σ-fixed class [b] and any x ∈ K′^×.
* **Ramification.** Hasse–Arf holds for abelian extensions with trivial residue extension. Suppose every cyclic degree-p subfield K′ ⊆ D is classically totally ramified with ramification break u(K′). Suppose also that every cyclic degree-p extension of K′ inside C_D(K′) is classically totally ramified with break ℓ satisfying ℓ > u(K′) and ℓ ≢ u(K′) (mod p). Then D has no Galois maximal subfield.
* **Necessary structure.** The residue field must be imperfect. For each K′, write C = C_D(K′). Either C contains no cyclic degree-p subfield over K′, or e_C = p, C̄ is purely inseparable of degree p over K̄′, and gr(C) is commutative.
* **Candidate invariant.** Kato's Swan conductor of [D ⊗_k K′]. Unverified is that sw([C]) ≤ ℓ, with equality for nondegenerate Artin–Schreier subfields.
* **Not available.** The roots-of-unity obstruction, Coyette's transfer lemma, and cyclicity of degree-p p-algebras.

**Route to a potential counterexample.** Construct a complete discretely valued field k of characteristic p with imperfect residue field, and a central division k-algebra D of degree p², such that:

1. D contains an Artin–Schreier element.
2. Every cyclic degree-p subfield K′ ⊆ D is classically totally ramified, with break u(K′).
3. Every cyclic degree-p extension of K′ inside C_D(K′) is classically totally ramified, with break ℓ > u(K′) and ℓ ≢ u(K′) (mod p).

Hasse–Arf then excludes every Galois maximal subfield. One valuation excludes both C_{p²} and C_p × C_p, where Coyette's construction needs two places.

Condition 3 would follow from sw([D ⊗ K′]) > u(K′) and sw([D ⊗ K′]) ≢ u(K′) (mod p), together with control of the degenerate subfields. Generic behavior under restriction gives sw([D ⊗ K′]) ≡ u(K′) (mod p), so D must have Swan data that degenerates on restriction to every cyclic K′ ⊆ D. Such a D would refute "Artin–Schreier element ⇒ crossed product" for division algebras of degree p², and it would settle Problem 2.2 negatively.
