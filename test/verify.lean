import Mathlib

-- The range check below evaluates 2^1074 and enumerates 2046 exponents in the
-- kernel, so raise the elaborator's exponentiation and recursion guards.
set_option maxRecDepth 100000
set_option exponentiation.threshold 5000

-- Exact rational value represented by binary significand f and exponent e.
def value (f : ℕ) (e : ℤ) : ℚ := f * 2 ^ e

-- One ULP for a regularly spaced value with exponent e.
def ulp (e : ℤ) : ℚ := 2 ^ e

-- Whether the exact rational result r rounds to the regularly spaced value
-- f · 2^e under round-to-nearest, ties-to-even.
def Roundtrips (f : ℕ) (e : ℤ) (r : ℚ) : Prop :=
  if f % 2 = 0 then
    |r - value f e| ≤ ulp e / 2
  else
    |r - value f e| < ulp e / 2

-- Whether f · 2^e is a regularly spaced normal binary64 value,
-- excluding powers of 2.
def Regular (f : ℕ) (e : ℤ) : Prop :=
  2 ^ 52 < f ∧ f < 2 ^ 53 ∧
    -1074 ≤ e ∧ e ≤ 971

-- Binary exponent of 10^k used to normalize its 128-bit significand.
def power10Exponent (k : ℤ) : ℤ :=
  if 0 ≤ k then
    Nat.log 2 (10 ^ k.toNat) + 1
  else
    -Nat.log 2 (10 ^ (-k).toNat)

-- Truncated 128-bit normalized binary significand of 10^k.
def power10Significand (k : ℤ) : ℕ :=
  ⌊(10 : ℚ) ^ k * 2 ^ (128 - power10Exponent k)⌋₊

-- Approximation of floor(e · log₁₀ 2) used as yy's decimal exponent.
def decimalExponent (e : ℤ) : ℤ :=
  e * 315_653 / 2 ^ 20

-- Shift chosen to align the binary exponent with the power of ten.
def decimalShift (e : ℤ) : ℕ :=
  Int.toNat (e + (-decimalExponent e * 217_707) / 2 ^ 16)

-- The 128-bit decimal significand ⌊f·2^(h+1)·⌊10^(-k)·2^128⌋ / 2^64⌋.
def scaledSignificand (f : ℕ) (e : ℤ) : ℕ :=
  let k := decimalExponent e
  let h := decimalShift e
  let p10 := power10Significand (-k)
  f * 2 ^ (h + 1) * p10 / 2 ^ 64

-- High and low 64-bit words of the decimal significand.
def sigHi (f : ℕ) (e : ℤ) : ℕ := scaledSignificand f e / 2 ^ 64
def sigLo (f : ℕ) (e : ℤ) : ℕ := scaledSignificand f e % 2 ^ 64

structure DecimalCandidates where
  k : ℤ
  decOne : ℕ
  roundU1 : Bool
  decTen : ℕ
  roundD0 : Bool
  roundU0 : Bool

def toDecimalCandidates (f : ℕ) (e : ℤ) : DecimalCandidates :=
  let k := decimalExponent e
  let h := decimalShift e

  let p10 := power10Significand (-k)
  let p10Hi := p10 / 2 ^ 64

  let sig := scaledSignificand f e
  let sigHi := sig / 2 ^ 64
  let sigLo := sig % 2 ^ 64

  let one := sigHi % 10
  let ten := sigHi - one
  let c := one * 2 ^ 60 + sigLo / 2 ^ 4
  let halfUlp := p10Hi / 2 ^ (4 - h)
  let t0 := 10 * 2 ^ 60
  let t1 := c + halfUlp

  let roundU1 : Bool :=
    if sigLo = 2 ^ 63 then
      sigHi % 2 = 1
    else
      2 ^ 63 < sigLo

  let roundD0 : Bool :=
    if halfUlp = c then
      f % 2 = 0
    else
      c < halfUlp

  let roundU0 : Bool :=
    if t1 + 1 = t0 then
      f % 2 = 0
    else if k = 0 ∧ t1 = t0 then
      f % 2 = 0
    else
      t0 ≤ t1

  {
    k := k
    roundU1 := roundU1
    decOne := sigHi + if roundU1 then 1 else 0
    decTen := ten + if roundU0 then 10 else 0
    roundD0 := roundD0
    roundU0 := roundU0
  }

-- Converts a regularly spaced binary floating-point value f · 2^e
-- to a decimal significand and exponent using yy's full path.
def toDecimal (f : ℕ) (e : ℤ) : ℕ × ℤ :=
  let c := toDecimalCandidates f e
  (if c.roundD0 || c.roundU0 then c.decTen else c.decOne, c.k)

-- The power-of-10 significand is the truncation of the exact scaled value.
theorem power10_significand_bounds (k : ℤ) :
    let x := (10 : ℚ) ^ k * 2 ^ (128 - power10Exponent k)
    (power10Significand k : ℚ) ≤ x ∧ x < power10Significand k + 1 := by
  dsimp [power10Significand]
  exact ⟨Nat.floor_le (by positivity), Nat.lt_floor_add_one _⟩

-- The shift used by yy's regular path is less than 4.
theorem decimal_shift_lt_four
    (f : ℕ) (e : ℤ)
    (h : Regular f e) :
    decimalShift e < 4 := by
  unfold decimalShift decimalExponent
  rcases h with ⟨_, _, hlo, hhi⟩
  omega

-- Exponent alignment over the binary64 exponent range.
theorem align_all :
    ∀ e ∈ Finset.Icc (-1074 : ℤ) 971,
      (decimalShift e : ℤ) + 1 - power10Exponent (-decimalExponent e) = e := by
  decide

-- Finite integer form of 2^e / 10^k ≥ 1 + 2⁻⁶² for e ≠ 0.
def marginHolds (e : ℤ) : Bool :=
  let k := decimalExponent e
  decide (2 ^ 1074 * ((2 ^ 62 + 1) * 10 ^ (k + 324).toNat)
            ≤ 2 ^ 62 * (2 ^ (e + 1074).toNat * 10 ^ 324))

theorem margin_all :
    ∀ e ∈ Finset.Icc (-1074 : ℤ) 971, e ≠ 0 → marginHolds e = true := by decide

-- Exponent alignment gives 2^(h+1)·2^(128-E) = 2^(e+128).
theorem aligned_pow (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    (2 : ℚ) ^ (decimalShift e + 1) *
        2 ^ (128 - power10Exponent (-decimalExponent e)) =
      2 ^ (e + 128) := by
  have halign :
      (decimalShift e : ℤ) + 1 -
          power10Exponent (-decimalExponent e) = e :=
    align_all e (by simpa [Finset.mem_Icc] using he)

  have hnp :
      (2 : ℚ) ^ (decimalShift e + 1) =
        (2 : ℚ) ^ ((decimalShift e : ℤ) + 1) := by
    rw [← zpow_natCast]
    congr 1

  rw [hnp, ← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
  congr 1
  omega

-- For e ≠ 0, 2^e / 10^k is at least 1 + 2⁻⁶².
theorem margin_lower (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) (he0 : e ≠ 0) :
    (1 : ℚ) + 1 / 2 ^ 62 ≤ 2 ^ e * (10 ^ decimalExponent e)⁻¹ := by
  have hb : 0 ≤ e + 1074 := by omega

  have hmz := margin_all e (by simpa [Finset.mem_Icc] using he) he0
  simp only [marginHolds, decide_eq_true_eq] at hmz

  set k := decimalExponent e
  have hk : 0 ≤ k + 324 := by
    show 0 ≤ decimalExponent e + 324; unfold decimalExponent; omega

  -- Cast the finite certificate to ℚ and expose the common factor.
  have hcert :
      (2 : ℚ) ^ (1074 : ℕ) * ((2 ^ 62 + 1) * 10 ^ (k + 324).toNat)
        ≤ 2 ^ (62 : ℕ) * (2 ^ (e + 1074).toNat * 10 ^ (324 : ℕ)) := by
    exact_mod_cast hmz
  simp only [← zpow_natCast, Int.toNat_of_nonneg hk, Int.toNat_of_nonneg hb,
    Nat.cast_ofNat] at hcert
  rw [zpow_add₀ (by norm_num : (10 : ℚ) ≠ 0) k 324,
      zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0) e 1074] at hcert

  have hfactor_pos : (0 : ℚ) < 2 ^ (1074 : ℤ) * 10 ^ (324 : ℤ) := by positivity
  have hc : ((2 : ℚ) ^ 62 + 1) * 10 ^ k ≤ 2 ^ 62 * 2 ^ e := by
    have hscaled :
        (2 ^ (1074 : ℤ) * 10 ^ (324 : ℤ)) * (((2 : ℚ) ^ (62 : ℤ) + 1) * 10 ^ k)
          ≤ (2 ^ (1074 : ℤ) * 10 ^ (324 : ℤ)) *
              ((2 : ℚ) ^ (62 : ℤ) * 2 ^ e) := by
      simpa only [mul_assoc, mul_comm, mul_left_comm] using hcert
    exact_mod_cast le_of_mul_le_mul_left hscaled hfactor_pos

   -- Normalize the bound by 2^62·10^k.
  have hp : (0 : ℚ) < 10 ^ k := by positivity
  rw [← div_eq_mul_inv, le_div_iff₀ hp]
  calc (1 + 1 / 2 ^ 62 : ℚ) * 10 ^ k
      = ((2 ^ 62 + 1) * 10 ^ k) / 2 ^ 62 := by field_simp
    _ ≤ (2 ^ 62 * 2 ^ e) / 2 ^ 62 := by gcongr
    _ = 2 ^ e := by field_simp

-- decOne rounds the fixed-point significand sig / 2^64 to the nearest integer,
-- so its error is at most 1/2.
theorem round_bound (f : ℕ) (e : ℤ) :
    |((toDecimalCandidates f e).decOne : ℚ)
        - (scaledSignificand f e : ℚ) / 2 ^ 64| ≤ 1 / 2 := by
  let c := toDecimalCandidates f e
  let hi := sigHi f e
  let lo := sigLo f e

  have hsplit :
      (scaledSignificand f e : ℚ) / 2 ^ 64 =
        (hi : ℚ) + lo / 2 ^ 64 := by
    have h := Nat.div_add_mod (scaledSignificand f e) (2 ^ 64)
    change 2 ^ 64 * hi + lo = scaledSignificand f e at h
    rw [← h]
    push_cast
    ring

  have hlo : lo < 2 ^ 64 := Nat.mod_lt _ (by norm_num)

  have hround :
      c.roundU1 =
        if lo = 2 ^ 63 then
          decide (hi % 2 = 1)
        else
          decide (2 ^ 63 < lo) := rfl

  have htrue : c.roundU1 = true → 2 ^ 63 ≤ lo := by
    intro h
    rw [hround] at h
    split at h
    · omega
    · rw [decide_eq_true_eq] at h; omega

  have hfalse : c.roundU1 = false → lo ≤ 2 ^ 63 := by
    intro h
    rw [hround] at h
    split at h
    · omega
    · simp only [decide_eq_false_iff_not, not_lt] at h; omega

  have hdec :
      c.decOne = hi + if c.roundU1 then 1 else 0 := rfl

  rw [hsplit, hdec]
  push_cast
  have hpos : (0 : ℚ) < 2 ^ 64 := by positivity
  have hpow : (2 : ℚ) ^ 64 = 2 * 2 ^ 63 := by norm_num
  have hle1 : (lo : ℚ) / 2 ^ 64 ≤ 1 :=
    (div_le_one hpos).2 (by exact_mod_cast hlo.le)
  have hge0 : (0 : ℚ) ≤ (lo : ℚ) / 2 ^ 64 := by positivity
  rw [abs_le]
  split_ifs with hround
  · -- Round up: 2^63 ≤ lo, so lo / 2^64 ≥ 1/2.
    have : (1 : ℚ) / 2 ≤ lo / 2 ^ 64 := by
      rw [le_div_iff₀ hpos]
      have : (2 : ℚ) ^ 63 ≤ lo := by exact_mod_cast htrue hround
      linarith [hpow]
    constructor <;> linarith
  · -- Round down: lo ≤ 2^63, so lo / 2^64 ≤ 1/2.
    simp only [Bool.not_eq_true] at hround
    have : (lo : ℚ) / 2 ^ 64 ≤ 1 / 2 := by
      rw [div_le_iff₀ hpos]
      have : (lo : ℚ) ≤ 2 ^ 63 := by exact_mod_cast hfalse hround
      linarith [hpow]
    constructor <;> linarith

-- The scaled error is the sum of power-of-ten truncation and the discarded
-- low word of the product.
theorem scaled_value_error_eq (f : ℕ) (e : ℤ) (hr : Regular f e) :
    let k := decimalExponent e
    let h := decimalShift e
    let p10 := power10Significand (-k)
    let exactP10 := (10 : ℚ) ^ (-k) * 2 ^ (128 - power10Exponent (-k))
    let fullProduct := f * 2 ^ (h + 1) * p10
    let x := value f e * (10 ^ k)⁻¹
    let q := (scaledSignificand f e : ℚ) / 2 ^ 64
    2 ^ 128 * (x - q) =
      (f : ℚ) * 2 ^ (h + 1) * (exactP10 - p10) +
        (fullProduct % 2 ^ 64 : ℕ) := by
  let k := decimalExponent e
  let h := decimalShift e
  let p10 := power10Significand (-k)
  let exactP10 : ℚ :=
    10 ^ (-k) * 2 ^ (128 - power10Exponent (-k))
  let fullProduct : ℕ := f * 2 ^ (h + 1) * p10
  let x := value f e * (10 ^ k)⁻¹
  let q := (scaledSignificand f e : ℚ) / 2 ^ 64
  change
    2 ^ 128 * (x - q) =
      ↑f * 2 ^ (h + 1) * (exactP10 - ↑p10) +
        (fullProduct % 2 ^ 64 : ℕ)

  have hsig : scaledSignificand f e = fullProduct / 2 ^ 64 := rfl

  have hdivmod :
      (fullProduct : ℚ) =
        2 ^ 64 * ((fullProduct / 2 ^ 64 : ℕ) : ℚ) +
          (fullProduct % 2 ^ 64 : ℕ) := by
    exact_mod_cast (Nat.div_add_mod fullProduct (2 ^ 64)).symm

  have hpow :
      (2 : ℚ) ^ 128 * 2 ^ e = 2 ^ (e + 128) := by
    rw [← zpow_natCast, ← zpow_add₀ (by norm_num)]
    congr 1
    push_cast
    ring

  have halign := aligned_pow e hr.2.2
  change
    (2 : ℚ) ^ (h + 1) *
      2 ^ (128 - power10Exponent (-k)) =
        2 ^ (e + 128)
    at halign

  have hx :
      (2 : ℚ) ^ 128 * x =
        ↑f * 2 ^ (h + 1) * exactP10 := by
    rw [show x = ↑f * 2 ^ e * 10 ^ (-k) by
      simp [x, value, ← zpow_neg]]
    simp only [exactP10]
    linear_combination
      (↑f * (10 : ℚ) ^ (-k)) * hpow - (↑f * (10 : ℚ) ^ (-k)) * halign

  have hq :
      (2 : ℚ) ^ 128 * q =
        fullProduct - (fullProduct % 2 ^ 64 : ℕ) := by
    have hqval : (2 : ℚ) ^ 128 * q =
        2 ^ 64 * ((fullProduct / 2 ^ 64 : ℕ) : ℚ) := by
      simp only [q, hsig]; ring
    rw [hqval]; linarith [hdivmod]

  have hproduct :
      (fullProduct : ℚ) = ↑f * 2 ^ (h + 1) * ↑p10 := by
    simp [fullProduct]

  rw [mul_sub, hx, hq, hproduct]
  ring

-- The exact scaled value x = f·2^e / 10^k lies just above
-- q = sig / 2^64. The gap x - q is nonnegative and below 2⁻⁶³,
-- accounting for power-of-ten truncation and the low-word floor; both errors
-- are dominated by the guard bits (f < 2^53 and shift + 1 ≤ 4).
theorem scaled_value_error_bound (f : ℕ) (e : ℤ) (hr : Regular f e) :
    let c := toDecimalCandidates f e
    let x := value f e * (10 ^ c.k)⁻¹
    let q := (scaledSignificand f e : ℚ) / 2 ^ 64
    0 ≤ x - q ∧ x - q < 1 / 2 ^ 63 := by
  let k := decimalExponent e
  let h := decimalShift e
  let p10 := power10Significand (-k)
  let exactP10 : ℚ := 10 ^ (-k) * 2 ^ (128 - power10Exponent (-k))
  let fullProduct : ℕ := f * 2 ^ (h + 1) * p10
  let x := value f e * (10 ^ k)⁻¹
  let q := (scaledSignificand f e : ℚ) / 2 ^ 64
  change 0 ≤ x - q ∧ x - q < 1 / 2 ^ 63

  -- The residual identity and power-of-ten truncation give the two error terms.
  have hresidual_eq := scaled_value_error_eq f e hr
  change
    2 ^ 128 * (x - q) =
      ↑f * 2 ^ (h + 1) * (exactP10 - ↑p10) + (fullProduct % 2 ^ 64 : ℕ)
    at hresidual_eq

  have hp10_bounds := power10_significand_bounds (-k)
  change (p10 : ℚ) ≤ exactP10 ∧ exactP10 < p10 + 1 at hp10_bounds
  obtain ⟨hp10_lo, hp10_hi⟩ := hp10_bounds

  -- 0 ≤ f·2^(h+1) ≤ 2^57 and 0 ≤ fullProduct mod 2^64 < 2^64.
  have hlt : h < 4 := by
    simpa [h] using decimal_shift_lt_four f e hr

  have h2h : (2 : ℚ) ^ (h + 1) ≤ 2 ^ 4 :=
    pow_le_pow_right₀ (by norm_num) (by omega)

  have hfp_nn : (0 : ℚ) ≤ ↑f * 2 ^ (h + 1) := by positivity

  have hfp_ub : (↑f : ℚ) * 2 ^ (h + 1) ≤ 2 ^ 57 := by
    calc
      (↑f : ℚ) * 2 ^ (h + 1) ≤ 2 ^ 53 * 2 ^ (h + 1) := by
        gcongr
        exact_mod_cast hr.2.1.le
      _ ≤ 2 ^ 53 * 2 ^ 4 := by gcongr
      _ = 2 ^ 57 := by norm_num

  have hr64 : ((fullProduct % 2 ^ 64 : ℕ) : ℚ) < 2 ^ 64 := by
    exact_mod_cast Nat.mod_lt fullProduct (by norm_num)

  have hr_nn : (0 : ℚ) ≤ ((fullProduct % 2 ^ 64 : ℕ) : ℚ) := by positivity
  have h2pos : (0 : ℚ) < 2 ^ 128 := by positivity

  constructor
  · nlinarith [hresidual_eq, hfp_nn, hp10_lo, hr_nn, h2pos]
  · nlinarith [hresidual_eq, hfp_nn, hfp_ub, hp10_lo, hp10_hi, hr64, h2pos]

-- The longer decimal candidate is strictly within half a scaled ULP of the
-- exact value. For e ≠ 0, combine round_bound and scaled_value_error_bound to
-- get a distance below 1/2 + 2⁻⁶³, then use margin_lower. For e = 0, we have
-- decOne = f = x exactly.
theorem dec_one_distance
    (f : ℕ) (e : ℤ)
    (h : Regular f e) :
    let c := toDecimalCandidates f e
    let x := value f e * (10 ^ c.k)⁻¹
    let u := ulp e * (10 ^ c.k)⁻¹
    |(c.decOne : ℚ) - x| < u / 2 := by
  have hr := round_bound f e
  obtain ⟨hn, hl⟩ := scaled_value_error_bound f e h
  set c := toDecimalCandidates f e
  set d : ℚ := (c.decOne : ℚ) with hd
  set x : ℚ := value f e * (10 ^ c.k)⁻¹ with hx
  set u : ℚ := ulp e * (10 ^ c.k)⁻¹ with hu
  show |d - x| < u / 2

  have key : |d - x| < 1 / 2 + 1 / 2 ^ 63 := by
    rw [abs_lt]
    have hr_bounds := abs_le.mp hr
    constructor <;> linarith

  by_cases he0 : e = 0
  · subst he0
    have hk0 : c.k = 0 := rfl
    have hx0 : x = (f : ℚ) := by
      rw [hx]
      simp only [value, hk0]
      norm_num
    have hnat : c.decOne = f := by
      rw [hx0, hd] at key
      obtain ⟨hbl, hbr⟩ := abs_lt.mp key
      have d1 : c.decOne < f + 1 := by
        exact_mod_cast (by linarith : (c.decOne : ℚ) < (f : ℚ) + 1)
      have d2 : f < c.decOne + 1 := by
        exact_mod_cast (by linarith : (f : ℚ) < (c.decOne : ℚ) + 1)
      omega
    have hupos : (0 : ℚ) < u := by
      rw [hu]
      simp only [ulp]
      positivity
    have hdx : d = x := by
      rw [hd, hx0, hnat]
    rw [hdx, sub_self, abs_zero]
    linarith [hupos]

  · have huge : (1 : ℚ) + 1 / 2 ^ 62 ≤ u := by
      rcases h with ⟨hlo, hhi, elo, ehi⟩
      rw [hu]
      simp only [ulp]
      exact margin_lower e ⟨elo, ehi⟩ he0
    linarith [key, huge]

-- The decimal significand produced by yy is within half a scaled ULP
-- of the exact value, with equality allowed only when f is even.
theorem decimal_significand_distance
    (f : ℕ) (e : ℤ)
    (h : Regular f e) :
    let (d, k) := toDecimal f e
    let x := value f e * 10 ^ (-k)
    let u := ulp e * 10 ^ (-k)
    if f % 2 = 0 then
      |d - x| ≤ u / 2
    else
      |d - x| < u / 2 := by
  let c := toDecimalCandidates f e

  rw [show toDecimal f e =
    (if c.roundD0 || c.roundU0 then c.decTen else c.decOne, c.k) from rfl]

  -- Match the goal's exact inverse notation to prevent type mismatch.
  let x := value f e * (10 ^ c.k)⁻¹
  let u := ulp e * (10 ^ c.k)⁻¹

  let InRange (dec : ℕ) : Prop :=
    if f % 2 = 0 then
      |dec - x| ≤ u / 2
    else
      |dec - x| < u / 2

  have hone : InRange c.decOne := by
    have hs := dec_one_distance f e h
    simp only [InRange, c, x, u]
    split_ifs <;> linarith [hs]
  have hten_d0 (hd0 : c.roundD0 = true) : InRange c.decTen := sorry
  have hten_u0 (hu0 : c.roundU0 = true) : InRange c.decTen := sorry

  cases hd0 : c.roundD0 <;>
    cases hu0 : c.roundU0 <;>
    simp_all [InRange, x, u, zpow_neg]

-- The decimal representation produced by yy round-trips to the original value.
theorem yy_roundtrips
    (f : ℕ) (e : ℤ)
    (h : Regular f e) :
    let (d, k) := toDecimal f e
    Roundtrips f e (d * 10 ^ k) := by
  rcases hdk : toDecimal f e with ⟨d, k⟩

  have hp : (0 : ℚ) < 10 ^ k := by positivity

  have hcancel : (10 : ℚ) ^ (-k) * 10 ^ k = 1 := by
    simpa only [zpow_neg] using inv_mul_cancel₀ (ne_of_gt hp)

  have hrescale_error :
      ((d : ℚ) - value f e * 10 ^ (-k)) * 10 ^ k =
        d * 10 ^ k - value f e := by
    rw [sub_mul, mul_assoc, hcancel, mul_one]

  have hscale :
      (ulp e * 10 ^ (-k) / 2) * 10 ^ k = ulp e / 2 := by
    rw [div_mul_eq_mul_div, mul_assoc, hcancel, mul_one]

  have hdist := decimal_significand_distance f e h

  simp only [Roundtrips]
  split_ifs with heven <;>
    rw [← hrescale_error, abs_mul, abs_of_pos hp, ← hscale]
  · exact mul_le_mul_of_nonneg_right
      (by simpa [hdk, heven] using hdist) (le_of_lt hp)
  · exact mul_lt_mul_of_pos_right
      (by simpa [hdk, heven] using hdist) hp
