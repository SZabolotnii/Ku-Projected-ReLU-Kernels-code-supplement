import Mathlib

/-!
# Projected-cumulant ReLU kernel: formalized statements

Statements of `theory/statements.tex` (pipeline stage 5), approved by the author and
frozen by `lean_gate.py snapshot` (`review/lean-snapshot.json`) on 2026-10-25. Helper
facts live as local `have`s inside each proof, since the frozen snapshot admits no new
declarations.

Standard normal density: `φ u = gaussianPDFReal 0 1 u`.
-/

set_option autoImplicit false

open Real MeasureTheory Set ProbabilityTheory

namespace ProjReLU

/-- Standard normal density. -/
noncomputable def φ (u : ℝ) : ℝ := gaussianPDFReal 0 1 u

/-- Standard normal distribution function. -/
noncomputable def Φ (u : ℝ) : ℝ := ∫ z in Iic u, φ z

/-- `J_k = ∫_0^∞ u^k φ(u) φ(c u) du` with `c = ρ / s`, `s = √(1-ρ²)`. -/
noncomputable def J (ρ : ℝ) (k : ℕ) : ℝ :=
  ∫ u in Ioi (0 : ℝ), u ^ k * φ u * φ (ρ / √(1 - ρ ^ 2) * u)

/-- `I_k = ∫_0^∞ u^k φ(u) Φ(c u) du` with `c = ρ / s`. -/
noncomputable def I (ρ : ℝ) (k : ℕ) : ℝ :=
  ∫ u in Ioi (0 : ℝ), u ^ k * φ u * Φ (ρ / √(1 - ρ ^ 2) * u)

/-! ## Lemma `lem:moments` -/

/-- `lem:moments`, closed form of `J_k`. -/
theorem J_closed (ρ : ℝ) (hρ : |ρ| < 1) (k : ℕ) :
    J ρ k = Real.Gamma (((k : ℝ) + 1) / 2) * (2 * (1 - ρ ^ 2)) ^ (((k : ℝ) + 1) / 2)
      / (4 * π) := by
  have φ_mul : ∀ u : ℝ, φ u * φ (ρ / √(1 - ρ ^ 2) * u) =
      (2 * π)⁻¹ * rexp (-(1 / (2 * (1 - ρ ^ 2))) * u ^ (2:ℝ)) := fun u => by
    have hs : 0 < 1 - ρ ^ 2 := by
      have : ρ ^ 2 < 1 := by
        have := sq_lt_sq' (by linarith [abs_lt.mp hρ]) (abs_lt.mp hρ).2
        simpa using this
      linarith
    have hsq : (√(1 - ρ ^ 2)) ^ 2 = 1 - ρ ^ 2 := Real.sq_sqrt hs.le
    have hsne : √(1 - ρ ^ 2) ≠ 0 := (Real.sqrt_pos.mpr hs).ne'
    simp only [φ, gaussianPDFReal, NNReal.coe_one, mul_one, sub_zero]
    rw [show u ^ (2:ℝ) = u ^ 2 by norm_cast]
    rw [mul_mul_mul_comm, ← Real.exp_add]
    congr 1
    · rw [← mul_inv, ← Real.sqrt_mul (by positivity), Real.sqrt_mul_self (by positivity)]
    · congr 1
      rw [mul_pow, div_pow, hsq]
      field_simp
      ring
  have hs : 0 < 1 - ρ ^ 2 := by
    have : ρ ^ 2 < 1 := by
      have := sq_lt_sq' (by linarith [abs_lt.mp hρ]) (abs_lt.mp hρ).2
      simpa using this
    linarith
  have hb : 0 < 1 / (2 * (1 - ρ ^ 2)) := by positivity
  unfold J
  have key : ∫ u in Ioi (0 : ℝ), u ^ k * φ u * φ (ρ / √(1 - ρ ^ 2) * u)
      = ∫ u in Ioi (0 : ℝ), (2 * π)⁻¹ * (u ^ (k:ℝ) * rexp (-(1 / (2 * (1 - ρ ^ 2))) * u ^ (2:ℝ))) := by
    refine setIntegral_congr_fun measurableSet_Ioi (fun u _ => ?_)
    rw [mul_assoc, φ_mul u, Real.rpow_natCast]
    ring
  rw [key, integral_const_mul,
    integral_rpow_mul_exp_neg_mul_rpow (by norm_num) (by have := (Nat.cast_nonneg k : (0:ℝ) ≤ k); linarith) hb]
  rw [show -((k:ℝ) + 1) / 2 = -(((k:ℝ) + 1) / 2) by ring, Real.rpow_neg hb.le,
    ← Real.inv_rpow hb.le]
  rw [show (1 / (2 * (1 - ρ ^ 2)))⁻¹ = 2 * (1 - ρ ^ 2) by field_simp]
  field_simp
  ring

theorem J_closed_witness : ∃ ρ : ℝ, |ρ| < 1 := ⟨0, by norm_num⟩

/-- `lem:moments`, `I_0 = 1/4 + arcsin ρ / (2π)`. -/
theorem I_zero (ρ : ℝ) (hρ : |ρ| < 1) :
    I ρ 0 = 1 / 4 + Real.arcsin ρ / (2 * π) := by
  have φ_eq : ∀ (z : ℝ), φ z = (√(2 * π))⁻¹ * rexp (-(1 / 2) * z ^ 2) := by
    intro z
    simp only [φ, gaussianPDFReal, NNReal.coe_one, mul_one, sub_zero]
    congr 2
    ring
  have φ_cont : Continuous φ := by
    have : φ = fun z => (√(2 * π))⁻¹ * rexp (-(1 / 2) * z ^ 2) := funext φ_eq
    rw [this]; fun_prop
  have φ_nonneg : ∀ (z : ℝ), 0 ≤ φ z := by
    intro z
    exact gaussianPDFReal_nonneg 0 1 z
  have φ_le : ∀ (z : ℝ), φ z ≤ φ 0 := by
    intro z
    rw [φ_eq, φ_eq]
    apply mul_le_mul_of_nonneg_left _ (by positivity)
    apply Real.exp_le_exp.mpr
    nlinarith [sq_nonneg z]
  have φ_even : ∀ (z : ℝ), φ (-z) = φ z := by
    intro z
    exact by rw [φ_eq, φ_eq, neg_sq]
  have φ_int : Integrable φ := by
    exact integrable_gaussianPDFReal 0 1
  have φ_total : ∫ z, φ z = 1 := by
    exact integral_gaussianPDFReal_eq_one 0 one_ne_zero
  have φ_deriv : ∀ (z : ℝ), HasDerivAt φ (-z * φ z) z := by
    intro z
    have : φ = fun z => (√(2 * π))⁻¹ * rexp (-(1 / 2) * z ^ 2) := funext φ_eq
    rw [this]
    have h := ((hasDerivAt_pow 2 z).const_mul (-(1 / 2 : ℝ))).exp.const_mul (√(2 * π))⁻¹
    convert h using 1
    push_cast; ring
  have half_Ioi : ∫ z in Ioi (0 : ℝ), φ z = 1 / 2 := by
    have h := integral_comp_abs (f := φ)
    have e : (fun x => φ |x|) = φ := funext fun x => by
      rcases abs_choice x with h | h <;> rw [h]; exact φ_even x
    rw [e, φ_total] at h
    linarith
  have Φ_zero : Φ 0 = 1 / 2 := by
    unfold Φ
    have h1 := intervalIntegral.integral_Iic_add_Ioi (b := (0 : ℝ)) φ_int.integrableOn
      φ_int.integrableOn
    rw [φ_total, half_Ioi] at h1
    linarith
  have Φ_nonneg : ∀ (u : ℝ), 0 ≤ Φ u := by
    intro u
    exact setIntegral_nonneg measurableSet_Iic (fun z _ => φ_nonneg z)
  have Φ_le_one : ∀ (u : ℝ), Φ u ≤ 1 := by
    intro u
    rw [← φ_total]
    exact setIntegral_le_integral φ_int (Filter.Eventually.of_forall φ_nonneg)
  have Φ_deriv : ∀ (u : ℝ), HasDerivAt Φ (φ u) u := by
    intro u
    have e : Φ = fun x => Φ 0 + ∫ z in (0 : ℝ)..x, φ z := funext fun x => by
      have := intervalIntegral.integral_Iic_sub_Iic (μ := volume) (f := φ) (a := (0 : ℝ)) (b := x)
        φ_int.integrableOn φ_int.integrableOn
      unfold Φ
      linarith
    rw [e]
    exact (intervalIntegral.integral_hasDerivAt_right (φ_cont.intervalIntegrable _ _)
      (φ_cont.stronglyMeasurableAtFilter _ _) φ_cont.continuousAt).const_add _
  have Φ_cont : Continuous Φ := by
    exact continuous_iff_continuousAt.mpr fun u => (Φ_deriv u).continuousAt
  have mom_integrable : ∀ (n : ℕ), IntegrableOn (fun z : ℝ => z ^ n * φ z) (Ioi 0) := by
    intro n
    have h : IntegrableOn
        (fun z : ℝ => (√(2 * π))⁻¹ * (z ^ (n : ℝ) * rexp (-(1 / 2) * z ^ 2))) (Ioi 0) :=
      (integrableOn_rpow_mul_exp_neg_mul_sq (b := 1 / 2) (by norm_num)
        (s := (n : ℝ)) (by have := (Nat.cast_nonneg n : (0:ℝ) ≤ n); linarith)).const_mul
        ((√(2 * π))⁻¹)
    refine IntegrableOn.congr_fun h (fun z _ => ?_) measurableSet_Ioi
    simp only [φ_eq, Real.rpow_natCast]
    ring
  have mom_tendsto : ∀ (n : ℕ), Filter.Tendsto (fun z : ℝ => z ^ n * φ z) Filter.atTop (nhds 0) := by
    intro n
    have h := (tendsto_rpow_mul_exp_neg_mul_atTop_nhds_zero ((n : ℝ) / 2) (1 / 2)
      (by norm_num)).comp (Filter.tendsto_pow_atTop (α := ℝ) two_ne_zero)
    have h2 := h.const_mul (√(2 * π))⁻¹
    rw [mul_zero] at h2
    refine h2.congr' ?_
    filter_upwards [Filter.eventually_gt_atTop 0] with z hz
    simp only [Function.comp, φ_eq]
    have e : (z ^ 2) ^ ((n : ℝ) / 2) = z ^ n := by
      rw [← Real.rpow_natCast z 2, ← Real.rpow_mul hz.le,
        show ((2 : ℕ) : ℝ) * ((n : ℝ) / 2) = (n : ℝ) by push_cast; ring, Real.rpow_natCast]
    rw [e]
    ring
  have mom_bdd_integrable : ∀ (n : ℕ) (g : ℝ → ℝ) (hg : Continuous g) (C : ℝ) (hC : ∀ z, |g z| ≤ C), IntegrableOn (fun z : ℝ => z ^ n * φ z * g z) (Ioi 0) := by
    intro n g hg C hC
    have h : IntegrableOn (fun z : ℝ => g z * (z ^ n * φ z)) (Ioi 0) :=
      Integrable.bdd_mul (mom_integrable n) hg.aestronglyMeasurable
        (Filter.Eventually.of_forall (fun z => by simpa [Real.norm_eq_abs] using hC z))
    exact IntegrableOn.congr_fun h (fun z _ => by ring) measurableSet_Ioi
  have ibp : ∀ (c : ℝ) (m : ℕ), ∫ x in Ioi (0 : ℝ), x ^ (m + 1) * φ x * Φ (c * x) = (0 : ℝ) ^ m * Φ 0 * φ 0 + ∫ x in Ioi (0 : ℝ), ((m : ℝ) * x ^ (m - 1) * Φ (c * x) + x ^ m * (φ (c * x) * (c * 1))) * φ x := by
    intro c m
    set u : ℝ → ℝ := fun x => x ^ m * Φ (c * x) with hu
    set v : ℝ → ℝ := fun x => -φ x with hv
    set u' : ℝ → ℝ := fun x => (m : ℝ) * x ^ (m - 1) * Φ (c * x) + x ^ m * (φ (c * x) * (c * 1))
      with hu'
    set v' : ℝ → ℝ := fun x => -(-x * φ x) with hv'
    have hΦc : Continuous fun x => Φ (c * x) := Φ_cont.comp (continuous_const.mul continuous_id)
    have hφc : Continuous fun x => φ (c * x) := φ_cont.comp (continuous_const.mul continuous_id)
    have hΦb : ∀ x, |Φ (c * x)| ≤ 1 := fun x => by
      rw [abs_of_nonneg (Φ_nonneg _)]; exact Φ_le_one _
    have hφb : ∀ x, |φ (c * x)| ≤ φ 0 := fun x => by
      rw [abs_of_nonneg (φ_nonneg _)]; exact φ_le _
    have hderu : ∀ x ∈ Ioi (0 : ℝ), HasDerivAt u (u' x) x := fun x _ => by
      have h1 := hasDerivAt_pow m x
      have h2 := (Φ_deriv (c * x)).comp x ((hasDerivAt_id x).const_mul c)
      exact (h1.mul h2).congr_deriv (by simp only [hu', Function.comp_apply])
    have hderv : ∀ x ∈ Ioi (0 : ℝ), HasDerivAt v (v' x) x := fun x _ => (φ_deriv x).neg
    have huv' : IntegrableOn (u * v') (Ioi 0) := by
      refine IntegrableOn.congr_fun (mom_bdd_integrable (m + 1) (fun x => Φ (c * x)) hΦc 1 hΦb)
        (fun x _ => ?_) measurableSet_Ioi
      simp only [Pi.mul_apply, hu, hv']; ring
    have hu'v : IntegrableOn (u' * v) (Ioi 0) := by
      have h1 := (mom_bdd_integrable (m - 1) (fun x => Φ (c * x)) hΦc 1 hΦb).const_mul (-(m : ℝ))
      have h2 := (mom_bdd_integrable m (fun x => φ (c * x)) hφc (φ 0) hφb).const_mul (-c)
      refine IntegrableOn.congr_fun (h1.add h2) (fun x _ => ?_) measurableSet_Ioi
      simp only [Pi.mul_apply, Pi.add_apply, hu', hv]; ring
    have hcont : Continuous (u * v) := by
      simp only [hu, hv]
      exact ((continuous_pow m).mul hΦc).mul φ_cont.neg
    have hzero : Filter.Tendsto (u * v) (nhdsWithin 0 (Ioi 0)) (nhds ((u * v) 0)) :=
      (hcont.tendsto 0).mono_left nhdsWithin_le_nhds
    have hinf : Filter.Tendsto (u * v) Filter.atTop (nhds 0) := by
      refine squeeze_zero_norm' ?_ (mom_tendsto m)
      filter_upwards [Filter.eventually_gt_atTop 0] with x hx
      simp only [Pi.mul_apply, hu, hv, Real.norm_eq_abs, abs_mul, abs_neg]
      rw [abs_of_nonneg (pow_nonneg hx.le m), abs_of_nonneg (φ_nonneg x),
        abs_of_nonneg (Φ_nonneg _)]
      have := Φ_le_one (c * x)
      have : 0 ≤ x ^ m * φ x := mul_nonneg (pow_nonneg hx.le m) (φ_nonneg x)
      nlinarith
    have key := integral_Ioi_mul_deriv_eq_deriv_mul hderu hderv huv' hu'v hzero hinf
    have lhs : ∫ x in Ioi (0 : ℝ), x ^ (m + 1) * φ x * Φ (c * x) = ∫ x in Ioi (0 : ℝ), u x * v' x :=
      setIntegral_congr_fun measurableSet_Ioi (fun x _ => by simp only [hu, hv']; ring)
    have rhs : ∫ x in Ioi (0 : ℝ), u' x * v x
        = -∫ x in Ioi (0 : ℝ), ((m : ℝ) * x ^ (m - 1) * Φ (c * x) + x ^ m * (φ (c * x) * (c * 1)))
              * φ x := by
      rw [← integral_neg]
      exact setIntegral_congr_fun measurableSet_Ioi (fun x _ => by simp only [hu', hv]; ring)
    rw [lhs, key, rhs]
    simp only [Pi.mul_apply, hu, hv, mul_zero]
    ring
  have F_deriv : ∀ (c : ℝ), HasDerivAt (fun x => ∫ a in Ioi (0 : ℝ), φ a * Φ (x * a)) (1 / (2 * π * (1 + c ^ 2))) c := by
    intro c
    have hb : Integrable (fun a : ℝ => φ 0 * (a ^ 1 * φ a)) (volume.restrict (Ioi 0)) :=
      (mom_integrable 1).const_mul (φ 0)
    have key := hasDerivAt_integral_of_dominated_loc_of_deriv_le (μ := volume.restrict (Ioi 0))
      (x₀ := c) (F := fun x a => φ a * Φ (x * a))
      (F' := fun x a => φ a * (φ (x * a) * a))
      (bound := fun a => φ 0 * |a ^ 1 * φ a|) one_pos
      (Filter.Eventually.of_forall fun x =>
        (φ_cont.mul (Φ_cont.comp (continuous_const.mul continuous_id))).aestronglyMeasurable)
      (by
        have := mom_bdd_integrable 0 (fun a => Φ (c * a))
          (Φ_cont.comp (continuous_const.mul continuous_id)) 1
          (fun a => by rw [abs_of_nonneg (Φ_nonneg _)]; exact Φ_le_one _)
        refine IntegrableOn.congr_fun this (fun a _ => ?_) measurableSet_Ioi
        simp only [pow_zero, one_mul, mul_comm c a])
      ((φ_cont.mul ((φ_cont.comp (continuous_const.mul continuous_id)).mul
        continuous_id)).aestronglyMeasurable)
      (Filter.Eventually.of_forall fun a x _ => by
        simp only [Real.norm_eq_abs, abs_mul, pow_one]
        rw [abs_of_nonneg (φ_nonneg a), abs_of_nonneg (φ_nonneg _)]
        have := φ_le (x * a)
        have h0 := φ_nonneg a
        have h1 := abs_nonneg a
        nlinarith [mul_nonneg h0 h1])
      (by
        have := hb.abs
        refine this.congr (Filter.Eventually.of_forall fun a => ?_)
        simp only [abs_mul, abs_of_nonneg (φ_nonneg 0)])
      (Filter.Eventually.of_forall fun a x _ => by
        have h := ((Φ_deriv (x * a)).comp x ((hasDerivAt_id x).mul_const a)).const_mul (φ a)
        simpa [Function.comp, one_mul] using h)
    refine key.2.congr_deriv ?_
    -- evaluate ∫_0^∞ a φ(a) φ(c a) da
    have e : ∫ a in Ioi (0 : ℝ), φ a * (φ (c * a) * a)
        = ∫ a in Ioi (0 : ℝ), (2 * π)⁻¹ * (a ^ (1 : ℝ) * rexp (-((1 + c ^ 2) / 2) * a ^ (2 : ℝ))) := by
      refine setIntegral_congr_fun measurableSet_Ioi (fun a _ => ?_)
      rw [φ_eq, φ_eq, Real.rpow_one, show a ^ (2 : ℝ) = a ^ 2 by norm_cast]
      have : (√(2 * π))⁻¹ * (√(2 * π))⁻¹ = (2 * π)⁻¹ := by
        rw [← mul_inv, Real.mul_self_sqrt (by positivity)]
      calc (√(2 * π))⁻¹ * rexp (-(1 / 2) * a ^ 2) * ((√(2 * π))⁻¹ * rexp (-(1 / 2) * (c * a) ^ 2) * a)
          = ((√(2 * π))⁻¹ * (√(2 * π))⁻¹) * (a * (rexp (-(1 / 2) * a ^ 2) *
              rexp (-(1 / 2) * (c * a) ^ 2))) := by ring
        _ = _ := by rw [this, ← Real.exp_add]; congr 3; ring
    rw [e, integral_const_mul, integral_rpow_mul_exp_neg_mul_rpow (by norm_num) (by norm_num)
      (by positivity)]
    have hpos : (0 : ℝ) < (1 + c ^ 2) / 2 := by positivity
    rw [show -((1 : ℝ) + 1) / 2 = -1 by norm_num, Real.rpow_neg_one,
      show ((1 : ℝ) + 1) / 2 = 1 by norm_num, Real.Gamma_one]
    field_simp
  set F := fun x => ∫ a in Ioi (0 : ℝ), φ a * Φ (x * a) with hF
  set G := fun x => F x - Real.arctan x / (2 * π) with hG
  have hGd : ∀ x, HasDerivAt G 0 x := fun x => by
    have h := (F_deriv x).sub ((Real.hasDerivAt_arctan x).div_const (2 * π))
    refine h.congr_deriv ?_
    field_simp
    ring
  have hconst := is_const_of_deriv_eq_zero (f := G) (fun x => (hGd x).differentiableAt)
    (fun x => (hGd x).deriv)
  have hF0 : F 0 = 1 / 4 := by
    simp only [hF, zero_mul, Φ_zero]
    rw [integral_mul_const, half_Ioi]; norm_num
  have hG0 : G 0 = 1 / 4 := by simp only [hG, hF0, Real.arctan_zero, zero_div, sub_zero]
  set c := ρ / √(1 - ρ ^ 2) with hc
  have hIF : I ρ 0 = F c := by
    simp only [I, hF, pow_zero, one_mul, ← hc, mul_comm c]
  have hmem : ρ ∈ Ioo (-(1 : ℝ)) 1 := ⟨by linarith [(abs_lt.mp hρ).1], (abs_lt.mp hρ).2⟩
  rw [hIF, Real.arcsin_eq_arctan hmem, ← hc]
  have := hconst c 0
  rw [hG0] at this
  simp only [hG] at this
  linarith

theorem I_zero_witness : ∃ ρ : ℝ, |ρ| < 1 := ⟨0, by norm_num⟩

/-- `lem:moments`, `I_1 = φ(0)/2 + c J_0`. -/
theorem I_one (ρ : ℝ) (hρ : |ρ| < 1) :
    I ρ 1 = φ 0 / 2 + ρ / √(1 - ρ ^ 2) * J ρ 0 := by
  have φ_eq : ∀ (z : ℝ), φ z = (√(2 * π))⁻¹ * rexp (-(1 / 2) * z ^ 2) := by
    intro z
    simp only [φ, gaussianPDFReal, NNReal.coe_one, mul_one, sub_zero]
    congr 2
    ring
  have φ_cont : Continuous φ := by
    have : φ = fun z => (√(2 * π))⁻¹ * rexp (-(1 / 2) * z ^ 2) := funext φ_eq
    rw [this]; fun_prop
  have φ_nonneg : ∀ (z : ℝ), 0 ≤ φ z := by
    intro z
    exact gaussianPDFReal_nonneg 0 1 z
  have φ_le : ∀ (z : ℝ), φ z ≤ φ 0 := by
    intro z
    rw [φ_eq, φ_eq]
    apply mul_le_mul_of_nonneg_left _ (by positivity)
    apply Real.exp_le_exp.mpr
    nlinarith [sq_nonneg z]
  have φ_even : ∀ (z : ℝ), φ (-z) = φ z := by
    intro z
    exact by rw [φ_eq, φ_eq, neg_sq]
  have φ_int : Integrable φ := by
    exact integrable_gaussianPDFReal 0 1
  have φ_total : ∫ z, φ z = 1 := by
    exact integral_gaussianPDFReal_eq_one 0 one_ne_zero
  have φ_deriv : ∀ (z : ℝ), HasDerivAt φ (-z * φ z) z := by
    intro z
    have : φ = fun z => (√(2 * π))⁻¹ * rexp (-(1 / 2) * z ^ 2) := funext φ_eq
    rw [this]
    have h := ((hasDerivAt_pow 2 z).const_mul (-(1 / 2 : ℝ))).exp.const_mul (√(2 * π))⁻¹
    convert h using 1
    push_cast; ring
  have half_Ioi : ∫ z in Ioi (0 : ℝ), φ z = 1 / 2 := by
    have h := integral_comp_abs (f := φ)
    have e : (fun x => φ |x|) = φ := funext fun x => by
      rcases abs_choice x with h | h <;> rw [h]; exact φ_even x
    rw [e, φ_total] at h
    linarith
  have Φ_zero : Φ 0 = 1 / 2 := by
    unfold Φ
    have h1 := intervalIntegral.integral_Iic_add_Ioi (b := (0 : ℝ)) φ_int.integrableOn
      φ_int.integrableOn
    rw [φ_total, half_Ioi] at h1
    linarith
  have Φ_nonneg : ∀ (u : ℝ), 0 ≤ Φ u := by
    intro u
    exact setIntegral_nonneg measurableSet_Iic (fun z _ => φ_nonneg z)
  have Φ_le_one : ∀ (u : ℝ), Φ u ≤ 1 := by
    intro u
    rw [← φ_total]
    exact setIntegral_le_integral φ_int (Filter.Eventually.of_forall φ_nonneg)
  have Φ_deriv : ∀ (u : ℝ), HasDerivAt Φ (φ u) u := by
    intro u
    have e : Φ = fun x => Φ 0 + ∫ z in (0 : ℝ)..x, φ z := funext fun x => by
      have := intervalIntegral.integral_Iic_sub_Iic (μ := volume) (f := φ) (a := (0 : ℝ)) (b := x)
        φ_int.integrableOn φ_int.integrableOn
      unfold Φ
      linarith
    rw [e]
    exact (intervalIntegral.integral_hasDerivAt_right (φ_cont.intervalIntegrable _ _)
      (φ_cont.stronglyMeasurableAtFilter _ _) φ_cont.continuousAt).const_add _
  have Φ_cont : Continuous Φ := by
    exact continuous_iff_continuousAt.mpr fun u => (Φ_deriv u).continuousAt
  have mom_integrable : ∀ (n : ℕ), IntegrableOn (fun z : ℝ => z ^ n * φ z) (Ioi 0) := by
    intro n
    have h : IntegrableOn
        (fun z : ℝ => (√(2 * π))⁻¹ * (z ^ (n : ℝ) * rexp (-(1 / 2) * z ^ 2))) (Ioi 0) :=
      (integrableOn_rpow_mul_exp_neg_mul_sq (b := 1 / 2) (by norm_num)
        (s := (n : ℝ)) (by have := (Nat.cast_nonneg n : (0:ℝ) ≤ n); linarith)).const_mul
        ((√(2 * π))⁻¹)
    refine IntegrableOn.congr_fun h (fun z _ => ?_) measurableSet_Ioi
    simp only [φ_eq, Real.rpow_natCast]
    ring
  have mom_tendsto : ∀ (n : ℕ), Filter.Tendsto (fun z : ℝ => z ^ n * φ z) Filter.atTop (nhds 0) := by
    intro n
    have h := (tendsto_rpow_mul_exp_neg_mul_atTop_nhds_zero ((n : ℝ) / 2) (1 / 2)
      (by norm_num)).comp (Filter.tendsto_pow_atTop (α := ℝ) two_ne_zero)
    have h2 := h.const_mul (√(2 * π))⁻¹
    rw [mul_zero] at h2
    refine h2.congr' ?_
    filter_upwards [Filter.eventually_gt_atTop 0] with z hz
    simp only [Function.comp, φ_eq]
    have e : (z ^ 2) ^ ((n : ℝ) / 2) = z ^ n := by
      rw [← Real.rpow_natCast z 2, ← Real.rpow_mul hz.le,
        show ((2 : ℕ) : ℝ) * ((n : ℝ) / 2) = (n : ℝ) by push_cast; ring, Real.rpow_natCast]
    rw [e]
    ring
  have mom_bdd_integrable : ∀ (n : ℕ) (g : ℝ → ℝ) (hg : Continuous g) (C : ℝ) (hC : ∀ z, |g z| ≤ C), IntegrableOn (fun z : ℝ => z ^ n * φ z * g z) (Ioi 0) := by
    intro n g hg C hC
    have h : IntegrableOn (fun z : ℝ => g z * (z ^ n * φ z)) (Ioi 0) :=
      Integrable.bdd_mul (mom_integrable n) hg.aestronglyMeasurable
        (Filter.Eventually.of_forall (fun z => by simpa [Real.norm_eq_abs] using hC z))
    exact IntegrableOn.congr_fun h (fun z _ => by ring) measurableSet_Ioi
  have ibp : ∀ (c : ℝ) (m : ℕ), ∫ x in Ioi (0 : ℝ), x ^ (m + 1) * φ x * Φ (c * x) = (0 : ℝ) ^ m * Φ 0 * φ 0 + ∫ x in Ioi (0 : ℝ), ((m : ℝ) * x ^ (m - 1) * Φ (c * x) + x ^ m * (φ (c * x) * (c * 1))) * φ x := by
    intro c m
    set u : ℝ → ℝ := fun x => x ^ m * Φ (c * x) with hu
    set v : ℝ → ℝ := fun x => -φ x with hv
    set u' : ℝ → ℝ := fun x => (m : ℝ) * x ^ (m - 1) * Φ (c * x) + x ^ m * (φ (c * x) * (c * 1))
      with hu'
    set v' : ℝ → ℝ := fun x => -(-x * φ x) with hv'
    have hΦc : Continuous fun x => Φ (c * x) := Φ_cont.comp (continuous_const.mul continuous_id)
    have hφc : Continuous fun x => φ (c * x) := φ_cont.comp (continuous_const.mul continuous_id)
    have hΦb : ∀ x, |Φ (c * x)| ≤ 1 := fun x => by
      rw [abs_of_nonneg (Φ_nonneg _)]; exact Φ_le_one _
    have hφb : ∀ x, |φ (c * x)| ≤ φ 0 := fun x => by
      rw [abs_of_nonneg (φ_nonneg _)]; exact φ_le _
    have hderu : ∀ x ∈ Ioi (0 : ℝ), HasDerivAt u (u' x) x := fun x _ => by
      have h1 := hasDerivAt_pow m x
      have h2 := (Φ_deriv (c * x)).comp x ((hasDerivAt_id x).const_mul c)
      exact (h1.mul h2).congr_deriv (by simp only [hu', Function.comp_apply])
    have hderv : ∀ x ∈ Ioi (0 : ℝ), HasDerivAt v (v' x) x := fun x _ => (φ_deriv x).neg
    have huv' : IntegrableOn (u * v') (Ioi 0) := by
      refine IntegrableOn.congr_fun (mom_bdd_integrable (m + 1) (fun x => Φ (c * x)) hΦc 1 hΦb)
        (fun x _ => ?_) measurableSet_Ioi
      simp only [Pi.mul_apply, hu, hv']; ring
    have hu'v : IntegrableOn (u' * v) (Ioi 0) := by
      have h1 := (mom_bdd_integrable (m - 1) (fun x => Φ (c * x)) hΦc 1 hΦb).const_mul (-(m : ℝ))
      have h2 := (mom_bdd_integrable m (fun x => φ (c * x)) hφc (φ 0) hφb).const_mul (-c)
      refine IntegrableOn.congr_fun (h1.add h2) (fun x _ => ?_) measurableSet_Ioi
      simp only [Pi.mul_apply, Pi.add_apply, hu', hv]; ring
    have hcont : Continuous (u * v) := by
      simp only [hu, hv]
      exact ((continuous_pow m).mul hΦc).mul φ_cont.neg
    have hzero : Filter.Tendsto (u * v) (nhdsWithin 0 (Ioi 0)) (nhds ((u * v) 0)) :=
      (hcont.tendsto 0).mono_left nhdsWithin_le_nhds
    have hinf : Filter.Tendsto (u * v) Filter.atTop (nhds 0) := by
      refine squeeze_zero_norm' ?_ (mom_tendsto m)
      filter_upwards [Filter.eventually_gt_atTop 0] with x hx
      simp only [Pi.mul_apply, hu, hv, Real.norm_eq_abs, abs_mul, abs_neg]
      rw [abs_of_nonneg (pow_nonneg hx.le m), abs_of_nonneg (φ_nonneg x),
        abs_of_nonneg (Φ_nonneg _)]
      have := Φ_le_one (c * x)
      have : 0 ≤ x ^ m * φ x := mul_nonneg (pow_nonneg hx.le m) (φ_nonneg x)
      nlinarith
    have key := integral_Ioi_mul_deriv_eq_deriv_mul hderu hderv huv' hu'v hzero hinf
    have lhs : ∫ x in Ioi (0 : ℝ), x ^ (m + 1) * φ x * Φ (c * x) = ∫ x in Ioi (0 : ℝ), u x * v' x :=
      setIntegral_congr_fun measurableSet_Ioi (fun x _ => by simp only [hu, hv']; ring)
    have rhs : ∫ x in Ioi (0 : ℝ), u' x * v x
        = -∫ x in Ioi (0 : ℝ), ((m : ℝ) * x ^ (m - 1) * Φ (c * x) + x ^ m * (φ (c * x) * (c * 1)))
              * φ x := by
      rw [← integral_neg]
      exact setIntegral_congr_fun measurableSet_Ioi (fun x _ => by simp only [hu', hv]; ring)
    rw [lhs, key, rhs]
    simp only [Pi.mul_apply, hu, hv, mul_zero]
    ring
  have F_deriv : ∀ (c : ℝ), HasDerivAt (fun x => ∫ a in Ioi (0 : ℝ), φ a * Φ (x * a)) (1 / (2 * π * (1 + c ^ 2))) c := by
    intro c
    have hb : Integrable (fun a : ℝ => φ 0 * (a ^ 1 * φ a)) (volume.restrict (Ioi 0)) :=
      (mom_integrable 1).const_mul (φ 0)
    have key := hasDerivAt_integral_of_dominated_loc_of_deriv_le (μ := volume.restrict (Ioi 0))
      (x₀ := c) (F := fun x a => φ a * Φ (x * a))
      (F' := fun x a => φ a * (φ (x * a) * a))
      (bound := fun a => φ 0 * |a ^ 1 * φ a|) one_pos
      (Filter.Eventually.of_forall fun x =>
        (φ_cont.mul (Φ_cont.comp (continuous_const.mul continuous_id))).aestronglyMeasurable)
      (by
        have := mom_bdd_integrable 0 (fun a => Φ (c * a))
          (Φ_cont.comp (continuous_const.mul continuous_id)) 1
          (fun a => by rw [abs_of_nonneg (Φ_nonneg _)]; exact Φ_le_one _)
        refine IntegrableOn.congr_fun this (fun a _ => ?_) measurableSet_Ioi
        simp only [pow_zero, one_mul, mul_comm c a])
      ((φ_cont.mul ((φ_cont.comp (continuous_const.mul continuous_id)).mul
        continuous_id)).aestronglyMeasurable)
      (Filter.Eventually.of_forall fun a x _ => by
        simp only [Real.norm_eq_abs, abs_mul, pow_one]
        rw [abs_of_nonneg (φ_nonneg a), abs_of_nonneg (φ_nonneg _)]
        have := φ_le (x * a)
        have h0 := φ_nonneg a
        have h1 := abs_nonneg a
        nlinarith [mul_nonneg h0 h1])
      (by
        have := hb.abs
        refine this.congr (Filter.Eventually.of_forall fun a => ?_)
        simp only [abs_mul, abs_of_nonneg (φ_nonneg 0)])
      (Filter.Eventually.of_forall fun a x _ => by
        have h := ((Φ_deriv (x * a)).comp x ((hasDerivAt_id x).mul_const a)).const_mul (φ a)
        simpa [Function.comp, one_mul] using h)
    refine key.2.congr_deriv ?_
    -- evaluate ∫_0^∞ a φ(a) φ(c a) da
    have e : ∫ a in Ioi (0 : ℝ), φ a * (φ (c * a) * a)
        = ∫ a in Ioi (0 : ℝ), (2 * π)⁻¹ * (a ^ (1 : ℝ) * rexp (-((1 + c ^ 2) / 2) * a ^ (2 : ℝ))) := by
      refine setIntegral_congr_fun measurableSet_Ioi (fun a _ => ?_)
      rw [φ_eq, φ_eq, Real.rpow_one, show a ^ (2 : ℝ) = a ^ 2 by norm_cast]
      have : (√(2 * π))⁻¹ * (√(2 * π))⁻¹ = (2 * π)⁻¹ := by
        rw [← mul_inv, Real.mul_self_sqrt (by positivity)]
      calc (√(2 * π))⁻¹ * rexp (-(1 / 2) * a ^ 2) * ((√(2 * π))⁻¹ * rexp (-(1 / 2) * (c * a) ^ 2) * a)
          = ((√(2 * π))⁻¹ * (√(2 * π))⁻¹) * (a * (rexp (-(1 / 2) * a ^ 2) *
              rexp (-(1 / 2) * (c * a) ^ 2))) := by ring
        _ = _ := by rw [this, ← Real.exp_add]; congr 3; ring
    rw [e, integral_const_mul, integral_rpow_mul_exp_neg_mul_rpow (by norm_num) (by norm_num)
      (by positivity)]
    have hpos : (0 : ℝ) < (1 + c ^ 2) / 2 := by positivity
    rw [show -((1 : ℝ) + 1) / 2 = -1 by norm_num, Real.rpow_neg_one,
      show ((1 : ℝ) + 1) / 2 = 1 by norm_num, Real.Gamma_one]
    field_simp
  set c := ρ / √(1 - ρ ^ 2)
  have h := ibp c 0
  simp only [zero_add, pow_zero, Nat.cast_zero, zero_mul, one_mul, mul_one] at h
  unfold I J
  rw [show (fun u : ℝ => u ^ 1 * φ u * Φ (c * u)) = fun u => u ^ (0 + 1) * φ u * Φ (c * u) from rfl]
  simp only [zero_add] at h ⊢
  rw [h, Φ_zero, ← integral_const_mul]
  congr 1
  · ring
  · exact setIntegral_congr_fun measurableSet_Ioi (fun u _ => by ring)

theorem I_one_witness : ∃ ρ : ℝ, |ρ| < 1 := ⟨0, by norm_num⟩

/-- `lem:moments`, recurrence `I_k = (k-1) I_{k-2} + c J_{k-1}` for `k ≥ 2`. -/
theorem I_rec (ρ : ℝ) (hρ : |ρ| < 1) (k : ℕ) (hk : 2 ≤ k) :
    I ρ k = ((k : ℝ) - 1) * I ρ (k - 2) + ρ / √(1 - ρ ^ 2) * J ρ (k - 1) := by
  have φ_eq : ∀ (z : ℝ), φ z = (√(2 * π))⁻¹ * rexp (-(1 / 2) * z ^ 2) := by
    intro z
    simp only [φ, gaussianPDFReal, NNReal.coe_one, mul_one, sub_zero]
    congr 2
    ring
  have φ_cont : Continuous φ := by
    have : φ = fun z => (√(2 * π))⁻¹ * rexp (-(1 / 2) * z ^ 2) := funext φ_eq
    rw [this]; fun_prop
  have φ_nonneg : ∀ (z : ℝ), 0 ≤ φ z := by
    intro z
    exact gaussianPDFReal_nonneg 0 1 z
  have φ_le : ∀ (z : ℝ), φ z ≤ φ 0 := by
    intro z
    rw [φ_eq, φ_eq]
    apply mul_le_mul_of_nonneg_left _ (by positivity)
    apply Real.exp_le_exp.mpr
    nlinarith [sq_nonneg z]
  have φ_even : ∀ (z : ℝ), φ (-z) = φ z := by
    intro z
    exact by rw [φ_eq, φ_eq, neg_sq]
  have φ_int : Integrable φ := by
    exact integrable_gaussianPDFReal 0 1
  have φ_total : ∫ z, φ z = 1 := by
    exact integral_gaussianPDFReal_eq_one 0 one_ne_zero
  have φ_deriv : ∀ (z : ℝ), HasDerivAt φ (-z * φ z) z := by
    intro z
    have : φ = fun z => (√(2 * π))⁻¹ * rexp (-(1 / 2) * z ^ 2) := funext φ_eq
    rw [this]
    have h := ((hasDerivAt_pow 2 z).const_mul (-(1 / 2 : ℝ))).exp.const_mul (√(2 * π))⁻¹
    convert h using 1
    push_cast; ring
  have half_Ioi : ∫ z in Ioi (0 : ℝ), φ z = 1 / 2 := by
    have h := integral_comp_abs (f := φ)
    have e : (fun x => φ |x|) = φ := funext fun x => by
      rcases abs_choice x with h | h <;> rw [h]; exact φ_even x
    rw [e, φ_total] at h
    linarith
  have Φ_zero : Φ 0 = 1 / 2 := by
    unfold Φ
    have h1 := intervalIntegral.integral_Iic_add_Ioi (b := (0 : ℝ)) φ_int.integrableOn
      φ_int.integrableOn
    rw [φ_total, half_Ioi] at h1
    linarith
  have Φ_nonneg : ∀ (u : ℝ), 0 ≤ Φ u := by
    intro u
    exact setIntegral_nonneg measurableSet_Iic (fun z _ => φ_nonneg z)
  have Φ_le_one : ∀ (u : ℝ), Φ u ≤ 1 := by
    intro u
    rw [← φ_total]
    exact setIntegral_le_integral φ_int (Filter.Eventually.of_forall φ_nonneg)
  have Φ_deriv : ∀ (u : ℝ), HasDerivAt Φ (φ u) u := by
    intro u
    have e : Φ = fun x => Φ 0 + ∫ z in (0 : ℝ)..x, φ z := funext fun x => by
      have := intervalIntegral.integral_Iic_sub_Iic (μ := volume) (f := φ) (a := (0 : ℝ)) (b := x)
        φ_int.integrableOn φ_int.integrableOn
      unfold Φ
      linarith
    rw [e]
    exact (intervalIntegral.integral_hasDerivAt_right (φ_cont.intervalIntegrable _ _)
      (φ_cont.stronglyMeasurableAtFilter _ _) φ_cont.continuousAt).const_add _
  have Φ_cont : Continuous Φ := by
    exact continuous_iff_continuousAt.mpr fun u => (Φ_deriv u).continuousAt
  have mom_integrable : ∀ (n : ℕ), IntegrableOn (fun z : ℝ => z ^ n * φ z) (Ioi 0) := by
    intro n
    have h : IntegrableOn
        (fun z : ℝ => (√(2 * π))⁻¹ * (z ^ (n : ℝ) * rexp (-(1 / 2) * z ^ 2))) (Ioi 0) :=
      (integrableOn_rpow_mul_exp_neg_mul_sq (b := 1 / 2) (by norm_num)
        (s := (n : ℝ)) (by have := (Nat.cast_nonneg n : (0:ℝ) ≤ n); linarith)).const_mul
        ((√(2 * π))⁻¹)
    refine IntegrableOn.congr_fun h (fun z _ => ?_) measurableSet_Ioi
    simp only [φ_eq, Real.rpow_natCast]
    ring
  have mom_tendsto : ∀ (n : ℕ), Filter.Tendsto (fun z : ℝ => z ^ n * φ z) Filter.atTop (nhds 0) := by
    intro n
    have h := (tendsto_rpow_mul_exp_neg_mul_atTop_nhds_zero ((n : ℝ) / 2) (1 / 2)
      (by norm_num)).comp (Filter.tendsto_pow_atTop (α := ℝ) two_ne_zero)
    have h2 := h.const_mul (√(2 * π))⁻¹
    rw [mul_zero] at h2
    refine h2.congr' ?_
    filter_upwards [Filter.eventually_gt_atTop 0] with z hz
    simp only [Function.comp, φ_eq]
    have e : (z ^ 2) ^ ((n : ℝ) / 2) = z ^ n := by
      rw [← Real.rpow_natCast z 2, ← Real.rpow_mul hz.le,
        show ((2 : ℕ) : ℝ) * ((n : ℝ) / 2) = (n : ℝ) by push_cast; ring, Real.rpow_natCast]
    rw [e]
    ring
  have mom_bdd_integrable : ∀ (n : ℕ) (g : ℝ → ℝ) (hg : Continuous g) (C : ℝ) (hC : ∀ z, |g z| ≤ C), IntegrableOn (fun z : ℝ => z ^ n * φ z * g z) (Ioi 0) := by
    intro n g hg C hC
    have h : IntegrableOn (fun z : ℝ => g z * (z ^ n * φ z)) (Ioi 0) :=
      Integrable.bdd_mul (mom_integrable n) hg.aestronglyMeasurable
        (Filter.Eventually.of_forall (fun z => by simpa [Real.norm_eq_abs] using hC z))
    exact IntegrableOn.congr_fun h (fun z _ => by ring) measurableSet_Ioi
  have ibp : ∀ (c : ℝ) (m : ℕ), ∫ x in Ioi (0 : ℝ), x ^ (m + 1) * φ x * Φ (c * x) = (0 : ℝ) ^ m * Φ 0 * φ 0 + ∫ x in Ioi (0 : ℝ), ((m : ℝ) * x ^ (m - 1) * Φ (c * x) + x ^ m * (φ (c * x) * (c * 1))) * φ x := by
    intro c m
    set u : ℝ → ℝ := fun x => x ^ m * Φ (c * x) with hu
    set v : ℝ → ℝ := fun x => -φ x with hv
    set u' : ℝ → ℝ := fun x => (m : ℝ) * x ^ (m - 1) * Φ (c * x) + x ^ m * (φ (c * x) * (c * 1))
      with hu'
    set v' : ℝ → ℝ := fun x => -(-x * φ x) with hv'
    have hΦc : Continuous fun x => Φ (c * x) := Φ_cont.comp (continuous_const.mul continuous_id)
    have hφc : Continuous fun x => φ (c * x) := φ_cont.comp (continuous_const.mul continuous_id)
    have hΦb : ∀ x, |Φ (c * x)| ≤ 1 := fun x => by
      rw [abs_of_nonneg (Φ_nonneg _)]; exact Φ_le_one _
    have hφb : ∀ x, |φ (c * x)| ≤ φ 0 := fun x => by
      rw [abs_of_nonneg (φ_nonneg _)]; exact φ_le _
    have hderu : ∀ x ∈ Ioi (0 : ℝ), HasDerivAt u (u' x) x := fun x _ => by
      have h1 := hasDerivAt_pow m x
      have h2 := (Φ_deriv (c * x)).comp x ((hasDerivAt_id x).const_mul c)
      exact (h1.mul h2).congr_deriv (by simp only [hu', Function.comp_apply])
    have hderv : ∀ x ∈ Ioi (0 : ℝ), HasDerivAt v (v' x) x := fun x _ => (φ_deriv x).neg
    have huv' : IntegrableOn (u * v') (Ioi 0) := by
      refine IntegrableOn.congr_fun (mom_bdd_integrable (m + 1) (fun x => Φ (c * x)) hΦc 1 hΦb)
        (fun x _ => ?_) measurableSet_Ioi
      simp only [Pi.mul_apply, hu, hv']; ring
    have hu'v : IntegrableOn (u' * v) (Ioi 0) := by
      have h1 := (mom_bdd_integrable (m - 1) (fun x => Φ (c * x)) hΦc 1 hΦb).const_mul (-(m : ℝ))
      have h2 := (mom_bdd_integrable m (fun x => φ (c * x)) hφc (φ 0) hφb).const_mul (-c)
      refine IntegrableOn.congr_fun (h1.add h2) (fun x _ => ?_) measurableSet_Ioi
      simp only [Pi.mul_apply, Pi.add_apply, hu', hv]; ring
    have hcont : Continuous (u * v) := by
      simp only [hu, hv]
      exact ((continuous_pow m).mul hΦc).mul φ_cont.neg
    have hzero : Filter.Tendsto (u * v) (nhdsWithin 0 (Ioi 0)) (nhds ((u * v) 0)) :=
      (hcont.tendsto 0).mono_left nhdsWithin_le_nhds
    have hinf : Filter.Tendsto (u * v) Filter.atTop (nhds 0) := by
      refine squeeze_zero_norm' ?_ (mom_tendsto m)
      filter_upwards [Filter.eventually_gt_atTop 0] with x hx
      simp only [Pi.mul_apply, hu, hv, Real.norm_eq_abs, abs_mul, abs_neg]
      rw [abs_of_nonneg (pow_nonneg hx.le m), abs_of_nonneg (φ_nonneg x),
        abs_of_nonneg (Φ_nonneg _)]
      have := Φ_le_one (c * x)
      have : 0 ≤ x ^ m * φ x := mul_nonneg (pow_nonneg hx.le m) (φ_nonneg x)
      nlinarith
    have key := integral_Ioi_mul_deriv_eq_deriv_mul hderu hderv huv' hu'v hzero hinf
    have lhs : ∫ x in Ioi (0 : ℝ), x ^ (m + 1) * φ x * Φ (c * x) = ∫ x in Ioi (0 : ℝ), u x * v' x :=
      setIntegral_congr_fun measurableSet_Ioi (fun x _ => by simp only [hu, hv']; ring)
    have rhs : ∫ x in Ioi (0 : ℝ), u' x * v x
        = -∫ x in Ioi (0 : ℝ), ((m : ℝ) * x ^ (m - 1) * Φ (c * x) + x ^ m * (φ (c * x) * (c * 1)))
              * φ x := by
      rw [← integral_neg]
      exact setIntegral_congr_fun measurableSet_Ioi (fun x _ => by simp only [hu', hv]; ring)
    rw [lhs, key, rhs]
    simp only [Pi.mul_apply, hu, hv, mul_zero]
    ring
  have F_deriv : ∀ (c : ℝ), HasDerivAt (fun x => ∫ a in Ioi (0 : ℝ), φ a * Φ (x * a)) (1 / (2 * π * (1 + c ^ 2))) c := by
    intro c
    have hb : Integrable (fun a : ℝ => φ 0 * (a ^ 1 * φ a)) (volume.restrict (Ioi 0)) :=
      (mom_integrable 1).const_mul (φ 0)
    have key := hasDerivAt_integral_of_dominated_loc_of_deriv_le (μ := volume.restrict (Ioi 0))
      (x₀ := c) (F := fun x a => φ a * Φ (x * a))
      (F' := fun x a => φ a * (φ (x * a) * a))
      (bound := fun a => φ 0 * |a ^ 1 * φ a|) one_pos
      (Filter.Eventually.of_forall fun x =>
        (φ_cont.mul (Φ_cont.comp (continuous_const.mul continuous_id))).aestronglyMeasurable)
      (by
        have := mom_bdd_integrable 0 (fun a => Φ (c * a))
          (Φ_cont.comp (continuous_const.mul continuous_id)) 1
          (fun a => by rw [abs_of_nonneg (Φ_nonneg _)]; exact Φ_le_one _)
        refine IntegrableOn.congr_fun this (fun a _ => ?_) measurableSet_Ioi
        simp only [pow_zero, one_mul, mul_comm c a])
      ((φ_cont.mul ((φ_cont.comp (continuous_const.mul continuous_id)).mul
        continuous_id)).aestronglyMeasurable)
      (Filter.Eventually.of_forall fun a x _ => by
        simp only [Real.norm_eq_abs, abs_mul, pow_one]
        rw [abs_of_nonneg (φ_nonneg a), abs_of_nonneg (φ_nonneg _)]
        have := φ_le (x * a)
        have h0 := φ_nonneg a
        have h1 := abs_nonneg a
        nlinarith [mul_nonneg h0 h1])
      (by
        have := hb.abs
        refine this.congr (Filter.Eventually.of_forall fun a => ?_)
        simp only [abs_mul, abs_of_nonneg (φ_nonneg 0)])
      (Filter.Eventually.of_forall fun a x _ => by
        have h := ((Φ_deriv (x * a)).comp x ((hasDerivAt_id x).mul_const a)).const_mul (φ a)
        simpa [Function.comp, one_mul] using h)
    refine key.2.congr_deriv ?_
    -- evaluate ∫_0^∞ a φ(a) φ(c a) da
    have e : ∫ a in Ioi (0 : ℝ), φ a * (φ (c * a) * a)
        = ∫ a in Ioi (0 : ℝ), (2 * π)⁻¹ * (a ^ (1 : ℝ) * rexp (-((1 + c ^ 2) / 2) * a ^ (2 : ℝ))) := by
      refine setIntegral_congr_fun measurableSet_Ioi (fun a _ => ?_)
      rw [φ_eq, φ_eq, Real.rpow_one, show a ^ (2 : ℝ) = a ^ 2 by norm_cast]
      have : (√(2 * π))⁻¹ * (√(2 * π))⁻¹ = (2 * π)⁻¹ := by
        rw [← mul_inv, Real.mul_self_sqrt (by positivity)]
      calc (√(2 * π))⁻¹ * rexp (-(1 / 2) * a ^ 2) * ((√(2 * π))⁻¹ * rexp (-(1 / 2) * (c * a) ^ 2) * a)
          = ((√(2 * π))⁻¹ * (√(2 * π))⁻¹) * (a * (rexp (-(1 / 2) * a ^ 2) *
              rexp (-(1 / 2) * (c * a) ^ 2))) := by ring
        _ = _ := by rw [this, ← Real.exp_add]; congr 3; ring
    rw [e, integral_const_mul, integral_rpow_mul_exp_neg_mul_rpow (by norm_num) (by norm_num)
      (by positivity)]
    have hpos : (0 : ℝ) < (1 + c ^ 2) / 2 := by positivity
    rw [show -((1 : ℝ) + 1) / 2 = -1 by norm_num, Real.rpow_neg_one,
      show ((1 : ℝ) + 1) / 2 = 1 by norm_num, Real.Gamma_one]
    field_simp
  set c := ρ / √(1 - ρ ^ 2) with hc
  obtain ⟨m, rfl⟩ : ∃ m, k = m + 2 := ⟨k - 2, by omega⟩
  have h := ibp c (m + 1)
  have hΦc : Continuous fun x => Φ (c * x) := Φ_cont.comp (continuous_const.mul continuous_id)
  have hφc : Continuous fun x => φ (c * x) := φ_cont.comp (continuous_const.mul continuous_id)
  have hΦb : ∀ x, |Φ (c * x)| ≤ 1 := fun x => by
    rw [abs_of_nonneg (Φ_nonneg _)]; exact Φ_le_one _
  have hφb : ∀ x, |φ (c * x)| ≤ φ 0 := fun x => by
    rw [abs_of_nonneg (φ_nonneg _)]; exact φ_le _
  have i1 := (mom_bdd_integrable m (fun x => Φ (c * x)) hΦc 1 hΦb).const_mul ((m : ℝ) + 1)
  have i2 := (mom_bdd_integrable (m + 1) (fun x => φ (c * x)) hφc (φ 0) hφb).const_mul c
  unfold I J
  simp only [show m + 2 - 2 = m by omega, show m + 2 - 1 = m + 1 by omega,
    show m + 1 - 1 = m by omega] at h ⊢
  rw [show m + 2 = m + 1 + 1 by ring, h]
  simp only [ne_eq, Nat.add_eq_zero_iff, one_ne_zero, and_false, not_false_eq_true, zero_pow,
    zero_mul, zero_add]
  rw [← integral_const_mul, ← integral_const_mul]
  simp only [← hc]
  rw [show ((m + 1 + 1 : ℕ) : ℝ) - 1 = (m : ℝ) + 1 by push_cast; ring, ← integral_add i1 i2]
  refine setIntegral_congr_fun measurableSet_Ioi (fun u _ => ?_)
  push_cast
  ring

theorem I_rec_witness : ∃ (ρ : ℝ) (k : ℕ), |ρ| < 1 ∧ 2 ≤ k := ⟨0, 2, by norm_num, le_rfl⟩

/-! ## Theorem `thm:closed`, the Gaussian term -/

/-- `thm:closed`, `T_00 = ρ I_2 + s J_1` is the arc-cosine kernel
`(s + (π - arccos ρ) ρ) / (2π)`. -/
theorem T00_arccos (ρ : ℝ) (hρ : |ρ| < 1) :
    ρ * I ρ 2 + √(1 - ρ ^ 2) * J ρ 1
      = (√(1 - ρ ^ 2) + (π - Real.arccos ρ) * ρ) / (2 * π) := by
  have hs : 0 < 1 - ρ ^ 2 := by
    have : ρ ^ 2 < 1 := by
      have := sq_lt_sq' (by linarith [abs_lt.mp hρ]) (abs_lt.mp hρ).2
      simpa using this
    linarith
  have hsp : 0 < √(1 - ρ ^ 2) := Real.sqrt_pos.mpr hs
  have hsq : (√(1 - ρ ^ 2)) ^ 2 = 1 - ρ ^ 2 := Real.sq_sqrt hs.le
  have hJ1 : J ρ 1 = (1 - ρ ^ 2) / (2 * π) := by
    rw [J_closed ρ hρ 1]
    norm_num [Real.Gamma_one]
    field_simp
    ring
  rw [I_rec ρ hρ 2 le_rfl, I_zero ρ hρ, Real.arccos_eq_pi_div_two_sub_arcsin]
  norm_num
  rw [hJ1]
  field_simp
  nlinarith [hsq]

theorem T00_arccos_witness : ∃ ρ : ℝ, |ρ| < 1 := ⟨0, by norm_num⟩

/-! ## Proposition `prop:limits`, the one-dimensional moments at `ρ = 1` -/

/-- `t_p = ∫_0^∞ z² He_p(z) φ(z) dz` for `p = 0, 3, 4, 6`
(`He_3 = z³ - 3z`, `He_4 = z⁴ - 6z² + 3`, `He_6 = z⁶ - 15z⁴ + 45z² - 15`). -/
theorem limit_moments :
    (∫ z in Ioi (0 : ℝ), z ^ 2 * φ z) = 1 / 2 ∧
    (∫ z in Ioi (0 : ℝ), z ^ 2 * (z ^ 3 - 3 * z) * φ z) = 2 * φ 0 ∧
    (∫ z in Ioi (0 : ℝ), z ^ 2 * (z ^ 4 - 6 * z ^ 2 + 3) * φ z) = 0 ∧
    (∫ z in Ioi (0 : ℝ), z ^ 2 * (z ^ 6 - 15 * z ^ 4 + 45 * z ^ 2 - 15) * φ z) = 0 := by
  let M : ℕ → ℝ := fun n => ∫ z in Ioi (0 : ℝ), z ^ n * φ z
  have φ_eq : ∀ z : ℝ, φ z = (√(2 * π))⁻¹ * rexp (-(1 / 2) * z ^ 2) := fun z => by
    simp only [φ, gaussianPDFReal, NNReal.coe_one, mul_one, sub_zero]
    congr 2
    ring
  have mom_integrable : ∀ n : ℕ, IntegrableOn (fun z : ℝ => z ^ n * φ z) (Ioi 0) := fun n => by
    have h : IntegrableOn
        (fun z : ℝ => (√(2 * π))⁻¹ * (z ^ (n : ℝ) * rexp (-(1 / 2) * z ^ 2))) (Ioi 0) :=
      (integrableOn_rpow_mul_exp_neg_mul_sq (b := 1 / 2) (by norm_num)
        (s := (n : ℝ)) (by have := (Nat.cast_nonneg n : (0:ℝ) ≤ n); linarith)).const_mul
        ((√(2 * π))⁻¹)
    refine IntegrableOn.congr_fun h (fun z _ => ?_) measurableSet_Ioi
    simp only [φ_eq, Real.rpow_natCast]
    ring
  have M_eq : ∀ n : ℕ, M n = (√(2 * π))⁻¹ * ((1 / 2 : ℝ) ^ (-((n : ℝ) + 1) / 2) * (1 / 2) *
      Real.Gamma (((n : ℝ) + 1) / 2)) := fun n => by
    have key : ∫ z in Ioi (0 : ℝ), z ^ n * φ z
        = ∫ z in Ioi (0 : ℝ), (√(2 * π))⁻¹ * (z ^ (n : ℝ) * rexp (-(1 / 2) * z ^ (2 : ℝ))) := by
      refine setIntegral_congr_fun measurableSet_Ioi (fun z _ => ?_)
      rw [φ_eq, Real.rpow_natCast, show z ^ (2:ℝ) = z ^ 2 by norm_cast]
      ring
    show ∫ z in Ioi (0 : ℝ), z ^ n * φ z = _
    rw [key, integral_const_mul, integral_rpow_mul_exp_neg_mul_rpow (by norm_num)
      (by have := (Nat.cast_nonneg n : (0:ℝ) ≤ n); linarith) (by norm_num)]
  have M_succ_succ : ∀ n : ℕ, M (n + 2) = ((n : ℝ) + 1) * M n := fun n => by
    rw [M_eq, M_eq]
    have hpos : (0 : ℝ) < ((n : ℝ) + 1) / 2 := by positivity
    have hG : Real.Gamma ((((n + 2 : ℕ) : ℝ) + 1) / 2) = (((n : ℝ) + 1) / 2) *
        Real.Gamma (((n : ℝ) + 1) / 2) := by
      rw [show (((n + 2 : ℕ) : ℝ) + 1) / 2 = ((n : ℝ) + 1) / 2 + 1 by push_cast; ring]
      exact Real.Gamma_add_one hpos.ne'
    have hP : (1 / 2 : ℝ) ^ (-(((n + 2 : ℕ) : ℝ) + 1) / 2) =
        2 * (1 / 2 : ℝ) ^ (-((n : ℝ) + 1) / 2) := by
      rw [show -(((n + 2 : ℕ) : ℝ) + 1) / 2 = -((n : ℝ) + 1) / 2 + (-1) by push_cast; ring,
        Real.rpow_add (by norm_num), Real.rpow_neg_one]
      norm_num
      ring
    rw [hG, hP]
    ring
  have M_zero : M 0 = 1 / 2 := by
    rw [M_eq]
    simp only [Nat.cast_zero, zero_add]
    rw [Real.Gamma_one_half_eq]
    rw [show -(1 : ℝ) / 2 = -(1 / 2) by ring, Real.rpow_neg (by norm_num),
      ← Real.sqrt_eq_rpow, Real.sqrt_mul (by norm_num)]
    have h2 : (0 : ℝ) < √2 := by positivity
    have hp : (0 : ℝ) < √π := Real.sqrt_pos.mpr Real.pi_pos
    rw [show √(1 / 2 : ℝ) = (√2)⁻¹ by rw [one_div, Real.sqrt_inv]]
    field_simp
  have M_one : M 1 = φ 0 := by
    rw [M_eq, φ_eq]
    norm_num [Real.Gamma_one]
  have I := mom_integrable
  have h2 : M 2 = 1 / 2 := by
    rw [show (2 : ℕ) = 0 + 2 from rfl, M_succ_succ, M_zero]; norm_num
  have h3 : M 3 = 2 * φ 0 := by
    rw [show (3 : ℕ) = 1 + 2 from rfl, M_succ_succ, M_one]; norm_num
  have h4 : M 4 = 3 * M 2 := by rw [show (4 : ℕ) = 2 + 2 from rfl, M_succ_succ]; norm_num
  have h5 : M 5 = 4 * M 3 := by rw [show (5 : ℕ) = 3 + 2 from rfl, M_succ_succ]; norm_num
  have h6 : M 6 = 5 * M 4 := by rw [show (6 : ℕ) = 4 + 2 from rfl, M_succ_succ]; norm_num
  have h8 : M 8 = 7 * M 6 := by rw [show (8 : ℕ) = 6 + 2 from rfl, M_succ_succ]; norm_num
  refine ⟨h2, ?_, ?_, ?_⟩
  · have e : ∫ z in Ioi (0 : ℝ), z ^ 2 * (z ^ 3 - 3 * z) * φ z
        = ∫ z in Ioi (0 : ℝ), (z ^ 5 * φ z - 3 * (z ^ 3 * φ z)) :=
      setIntegral_congr_fun measurableSet_Ioi (fun z _ => by ring)
    rw [e, integral_sub (I 5) ((I 3).const_mul 3), integral_const_mul]
    change M 5 - 3 * M 3 = _
    rw [h5, h3]; ring
  · have e : ∫ z in Ioi (0 : ℝ), z ^ 2 * (z ^ 4 - 6 * z ^ 2 + 3) * φ z
        = ∫ z in Ioi (0 : ℝ), (z ^ 6 * φ z - 6 * (z ^ 4 * φ z) + 3 * (z ^ 2 * φ z)) :=
      setIntegral_congr_fun measurableSet_Ioi (fun z _ => by ring)
    rw [e, integral_add, integral_sub, integral_const_mul, integral_const_mul]
    all_goals first
      | exact I _
      | exact (I _).const_mul _
      | exact (I _).sub ((I _).const_mul _)
      | exact ((I _).sub ((I _).const_mul _)).add ((I _).const_mul _)
      | skip
    change M 6 - 6 * M 4 + 3 * M 2 = _
    rw [h6, h4]; ring
  · have e : ∫ z in Ioi (0 : ℝ), z ^ 2 * (z ^ 6 - 15 * z ^ 4 + 45 * z ^ 2 - 15) * φ z
        = ∫ z in Ioi (0 : ℝ), (z ^ 8 * φ z - 15 * (z ^ 6 * φ z) + 45 * (z ^ 4 * φ z)
            - 15 * (z ^ 2 * φ z)) :=
      setIntegral_congr_fun measurableSet_Ioi (fun z _ => by ring)
    rw [e, integral_sub, integral_add, integral_sub, integral_const_mul, integral_const_mul,
      integral_const_mul]
    all_goals first
      | exact I _
      | exact (I _).const_mul _
      | exact (I _).sub ((I _).const_mul _)
      | exact ((I _).sub ((I _).const_mul _)).add ((I _).const_mul _)
      | skip
    change M 8 - 15 * M 6 + 45 * M 4 - 15 * M 2 = _
    rw [h8, h6, h4]; ring

/-! ## Lemma `lem:cramer` -/

/-- `lem:cramer`: dense orthonormal `a, b` and a modulus bound `ψ` of the
characteristic function of `W₁` (`ψ ≤ 1`, `ψ τ ≤ η` for `|τ| ≥ β/√2`) give the
averaged Cramér bound for `X_j = √d (a_j, b_j) W_j`. -/
theorem dense_avg_cramer {d : ℕ} (hd : 0 < d) (a b : Fin d → ℝ) (c β η : ℝ)
    (ψ : ℝ → ℝ)
    (ha : ∑ j, a j ^ 2 = 1) (hb : ∑ j, b j ^ 2 = 1) (hab : ∑ j, a j * b j = 0)
    (hca : ∀ j, |a j| ≤ c / √(d : ℝ)) (hcb : ∀ j, |b j| ≤ c / √(d : ℝ))
    (hβ : 0 < β) (hψ1 : ∀ τ, ψ τ ≤ 1) (hψη : ∀ τ, β / √2 ≤ |τ| → ψ τ ≤ η)
    (t₁ t₂ : ℝ) (ht : β ≤ √(t₁ ^ 2 + t₂ ^ 2)) :
    (1 / (d : ℝ)) * ∑ j, ψ (√(d : ℝ) * (t₁ * a j + t₂ * b j))
      ≤ 1 - (1 - η) / (4 * c ^ 2) := by
  classical
  have hdR : (0 : ℝ) < d := by exact_mod_cast hd
  have hsd : 0 < √(d : ℝ) := Real.sqrt_pos.mpr hdR
  have hsd2 : (√(d : ℝ)) ^ 2 = d := Real.sq_sqrt hdR.le
  -- c > 0
  have hc : 0 < c := by
    by_contra hcn
    push_neg at hcn
    have hz : ∀ j, a j = 0 := fun j => by
      have := hca j
      have : c / √(d : ℝ) ≤ 0 := div_nonpos_of_nonpos_of_nonneg hcn hsd.le
      exact abs_eq_zero.mp (le_antisymm (by linarith [hca j]) (abs_nonneg _))
    simp [hz] at ha
  have hc2 : 0 < c ^ 2 := by positivity
  -- squared coordinate bounds
  have hsqa : ∀ j, a j ^ 2 ≤ c ^ 2 / d := fun j => by
    have h1 : |a j| ^ 2 ≤ (c / √(d : ℝ)) ^ 2 :=
      pow_le_pow_left₀ (abs_nonneg _) (hca j) 2
    rwa [sq_abs, div_pow, hsd2] at h1
  have hsqb : ∀ j, b j ^ 2 ≤ c ^ 2 / d := fun j => by
    have h1 : |b j| ^ 2 ≤ (c / √(d : ℝ)) ^ 2 :=
      pow_le_pow_left₀ (abs_nonneg _) (hcb j) 2
    rwa [sq_abs, div_pow, hsd2] at h1
  set n2 := t₁ ^ 2 + t₂ ^ 2 with hn2
  have hβ2 : β ^ 2 ≤ n2 := by
    have := pow_le_pow_left₀ hβ.le ht 2
    rwa [Real.sq_sqrt (by positivity)] at this
  have hn2pos : 0 < n2 := lt_of_lt_of_le (by positivity) hβ2
  set v : Fin d → ℝ := fun j => t₁ * a j + t₂ * b j with hv
  have hsumv : ∑ j, v j ^ 2 = n2 := by
    have e : ∀ j, v j ^ 2 = t₁ ^ 2 * a j ^ 2 + (2 * t₁ * t₂) * (a j * b j) + t₂ ^ 2 * b j ^ 2 :=
      fun j => by simp only [hv]; ring
    rw [Finset.sum_congr rfl (fun j _ => e j), Finset.sum_add_distrib, Finset.sum_add_distrib,
      ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum, ha, hb, hab]
    ring
  have hvle : ∀ j, v j ^ 2 ≤ n2 * (2 * c ^ 2 / d) := fun j => by
    have cs : v j ^ 2 ≤ n2 * (a j ^ 2 + b j ^ 2) := by
      simp only [hv, hn2]
      nlinarith [sq_nonneg (t₁ * b j - t₂ * a j)]
    have : a j ^ 2 + b j ^ 2 ≤ 2 * c ^ 2 / d := by
      have := hsqa j; have := hsqb j
      rw [show 2 * c ^ 2 / (d : ℝ) = c ^ 2 / d + c ^ 2 / d by ring]; linarith
    calc v j ^ 2 ≤ n2 * (a j ^ 2 + b j ^ 2) := cs
      _ ≤ n2 * (2 * c ^ 2 / d) := mul_le_mul_of_nonneg_left this hn2pos.le
  -- the large set
  set J := Finset.univ.filter (fun j : Fin d => n2 / (2 * d) ≤ v j ^ 2) with hJ
  set Jc := Finset.univ.filter (fun j : Fin d => ¬ (n2 / (2 * d) ≤ v j ^ 2)) with hJc
  have hcard : (J.card : ℝ) + Jc.card = d := by
    have := Finset.filter_card_add_filter_neg_card_eq_card
      (s := (Finset.univ : Finset (Fin d))) (fun j : Fin d => n2 / (2 * d) ≤ v j ^ 2)
    rw [Finset.card_univ, Fintype.card_fin] at this
    rw [hJ, hJc]
    exact_mod_cast this
  have hsplit : ∀ f : Fin d → ℝ, ∑ j, f j = ∑ j ∈ J, f j + ∑ j ∈ Jc, f j := fun f =>
    (Finset.sum_filter_add_sum_filter_not _ _ _).symm
  -- counting bound
  have hcount : (d : ℝ) / (4 * c ^ 2) ≤ J.card := by
    have h1 : ∑ j ∈ J, v j ^ 2 ≤ J.card * (n2 * (2 * c ^ 2 / d)) := by
      have := Finset.sum_le_card_nsmul J (fun j => v j ^ 2) _ (fun j _ => hvle j)
      simpa [nsmul_eq_mul] using this
    have h2 : ∑ j ∈ Jc, v j ^ 2 ≤ Jc.card * (n2 / (2 * d)) := by
      have := Finset.sum_le_card_nsmul Jc (fun j => v j ^ 2) _
        (fun j hj => by
          have := (Finset.mem_filter.mp hj).2
          exact (not_le.mp this).le)
      simpa [nsmul_eq_mul] using this
    have h3 : n2 ≤ J.card * (n2 * (2 * c ^ 2 / d)) + Jc.card * (n2 / (2 * d)) := by
      have := hsplit (fun j => v j ^ 2)
      linarith [hsumv]
    have hJc : (Jc.card : ℝ) ≤ d := by linarith [(Nat.cast_nonneg J.card : (0:ℝ) ≤ J.card)]
    have h4 : n2 ≤ n2 * ((J.card : ℝ) * (2 * c ^ 2 / d) + 1 / 2) := by
      have : (Jc.card : ℝ) * (n2 / (2 * d)) ≤ n2 / 2 := by
        rw [show n2 / 2 = d * (n2 / (2 * d)) by field_simp]
        exact mul_le_mul_of_nonneg_right hJc (by positivity)
      nlinarith
    have h5 : 1 ≤ (J.card : ℝ) * (2 * c ^ 2 / d) + 1 / 2 := by
      by_contra hlt; push_neg at hlt
      nlinarith
    rw [div_le_iff₀ (by positivity)]
    have : (J.card : ℝ) * (2 * c ^ 2 / d) * d = J.card * (2 * c ^ 2) := by field_simp
    nlinarith
  -- characteristic-function bound on J
  have hJψ : ∀ j ∈ J, ψ (√(d : ℝ) * v j) ≤ η := fun j hj => by
    apply hψη
    have hm := (Finset.mem_filter.mp hj).2
    have hsq : (β / √2) ^ 2 ≤ (√(d : ℝ) * v j) ^ 2 := by
      rw [div_pow, Real.sq_sqrt (by norm_num), mul_pow, hsd2]
      have : n2 / 2 ≤ d * v j ^ 2 := by
        have := mul_le_mul_of_nonneg_left hm hdR.le
        rwa [show (d : ℝ) * (n2 / (2 * d)) = n2 / 2 by field_simp] at this
      linarith
    rw [← sq_abs (√(d : ℝ) * v j)] at hsq
    exact (sq_le_sq₀ (by positivity) (abs_nonneg _)).mp hsq
  have hsumψ : ∑ j, ψ (√(d : ℝ) * v j) ≤ J.card * η + Jc.card := by
    rw [hsplit]
    have h1 : ∑ j ∈ J, ψ (√(d : ℝ) * v j) ≤ J.card * η := by
      have := Finset.sum_le_card_nsmul J (fun j => ψ (√(d : ℝ) * v j)) η hJψ
      simpa [nsmul_eq_mul] using this
    have h2 : ∑ j ∈ Jc, ψ (√(d : ℝ) * v j) ≤ Jc.card * 1 := by
      have := Finset.sum_le_card_nsmul Jc (fun j => ψ (√(d : ℝ) * v j)) 1 (fun j _ => hψ1 _)
      simpa [nsmul_eq_mul] using this
    linarith
  change (1 / (d : ℝ)) * ∑ j, ψ (√(d : ℝ) * v j) ≤ _
  rcases le_or_gt η 1 with hη | hη
  · have key : (1 / (d : ℝ)) * ∑ j, ψ (√(d : ℝ) * v j) ≤ 1 - (J.card / d) * (1 - η) := by
      have : (1 / (d : ℝ)) * ∑ j, ψ (√(d : ℝ) * v j) ≤ (1 / d) * (J.card * η + Jc.card) :=
        mul_le_mul_of_nonneg_left hsumψ (by positivity)
      calc _ ≤ (1 / (d : ℝ)) * (J.card * η + Jc.card) := this
        _ = 1 - (J.card / d) * (1 - η) := by
          rw [show (Jc.card : ℝ) = (d : ℝ) - (J.card : ℝ) by linarith]; field_simp; ring
    have hfrac : 1 / (4 * c ^ 2) ≤ (J.card : ℝ) / d := by
      rw [le_div_iff₀ hdR]
      calc 1 / (4 * c ^ 2) * (d : ℝ) = (d : ℝ) / (4 * c ^ 2) := by ring
        _ ≤ (J.card : ℝ) := hcount
    have : (1 - η) / (4 * c ^ 2) ≤ (J.card / d) * (1 - η) := by
      calc (1 - η) / (4 * c ^ 2) = (1 / (4 * c ^ 2)) * (1 - η) := by ring
        _ ≤ (J.card / d) * (1 - η) := mul_le_mul_of_nonneg_right hfrac (by linarith)
    linarith
  · have h1 : ∑ j, ψ (√(d : ℝ) * v j) ≤ d := by
      have := Finset.sum_le_card_nsmul Finset.univ (fun j => ψ (√(d : ℝ) * v j)) 1
        (fun j _ => hψ1 _)
      simpa [nsmul_eq_mul] using this
    have h2 : (1 / (d : ℝ)) * ∑ j, ψ (√(d : ℝ) * v j) ≤ 1 := by
      calc _ ≤ (1 / (d : ℝ)) * d := mul_le_mul_of_nonneg_left h1 (by positivity)
        _ = 1 := by field_simp
    have h3 : 0 ≤ -(1 - η) / (4 * c ^ 2) := div_nonneg (by linarith) (by positivity)
    have : 1 - (1 - η) / (4 * c ^ 2) = 1 + (-(1 - η) / (4 * c ^ 2)) := by ring
    linarith

theorem dense_avg_cramer_witness :
    ∃ (d : ℕ) (_ : 0 < d) (a b : Fin d → ℝ) (c β η : ℝ) (ψ : ℝ → ℝ)
      (_ : ∑ j, a j ^ 2 = 1) (_ : ∑ j, b j ^ 2 = 1) (_ : ∑ j, a j * b j = 0)
      (_ : ∀ j, |a j| ≤ c / √(d : ℝ)) (_ : ∀ j, |b j| ≤ c / √(d : ℝ))
      (_ : 0 < β) (_ : ∀ τ, ψ τ ≤ 1) (_ : ∀ τ, β / √2 ≤ |τ| → ψ τ ≤ η)
      (t₁ t₂ : ℝ), β ≤ √(t₁ ^ 2 + t₂ ^ 2) := by
  refine ⟨2, by norm_num, ![1, 0], ![0, 1], √2, 1, 1, fun _ => 1, ?_, ?_, ?_, ?_, ?_,
    by norm_num, fun _ => le_rfl, fun _ _ => le_rfl, 1, 0, ?_⟩
  · simp [Fin.sum_univ_two]
  · simp [Fin.sum_univ_two]
  · simp [Fin.sum_univ_two]
  · intro j; fin_cases j <;> simp
  · intro j; fin_cases j <;> simp
  · simp

end ProjReLU

/-! Axiom audit, read by `lean_gate.py` G4. -/
#print axioms ProjReLU.J_closed
#print axioms ProjReLU.J_closed_witness
#print axioms ProjReLU.I_zero
#print axioms ProjReLU.I_zero_witness
#print axioms ProjReLU.I_one
#print axioms ProjReLU.I_one_witness
#print axioms ProjReLU.I_rec
#print axioms ProjReLU.I_rec_witness
#print axioms ProjReLU.T00_arccos
#print axioms ProjReLU.T00_arccos_witness
#print axioms ProjReLU.limit_moments
#print axioms ProjReLU.dense_avg_cramer
#print axioms ProjReLU.dense_avg_cramer_witness
