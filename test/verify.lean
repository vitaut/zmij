import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.Tactic

-- The finite checks below enumerate 2046 exponents in the kernel; each one
-- raises the recursion guard locally, and the ones that evaluate 2^1074 also
-- raise the elaborator's exponentiation guard.
set_option maxRecDepth 100000

/-! ### The specification -/

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

/-! ### yy's conversion -/

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

/-! ### The truncated power of ten -/

-- The power-of-10 significand is the truncation of the exact scaled value.
theorem power10_significand_bounds (k : ℤ) :
    let x := (10 : ℚ) ^ k * 2 ^ (128 - power10Exponent k)
    (power10Significand k : ℚ) ≤ x ∧ x < power10Significand k + 1 := by
  dsimp [power10Significand]
  exact ⟨Nat.floor_le (by positivity), Nat.lt_floor_add_one _⟩

-- The power-of-ten significand is normalized: its top bit is set. This is
-- what makes power10Exponent an exponent for a 128-bit significand.
theorem power10_significand_lower (k : ℤ) :
    2 ^ 127 ≤ power10Significand k := by
  -- 2^(pe-1) ≤ 10^k by the defining property of Nat.log.
  have hlog : (2 : ℚ) ^ (power10Exponent k - 1) ≤ 10 ^ k := by
    unfold power10Exponent
    split_ifs with hk
    · have hpos : 10 ^ k.toNat ≠ 0 := by positivity
      have hnat : 2 ^ Nat.log 2 (10 ^ k.toNat) ≤ 10 ^ k.toNat :=
        Nat.pow_log_le_self 2 hpos
      have : ((2 : ℚ) ^ Nat.log 2 (10 ^ k.toNat)) ≤ ((10 : ℚ) ^ k.toNat) := by
        exact_mod_cast hnat
      simpa [add_sub_cancel_right, ← zpow_natCast (10 : ℚ) k.toNat,
        Int.toNat_of_nonneg hk] using this
    · set m := (-k).toNat with hm
      set l := Nat.log 2 (10 ^ m)
      have hq : ((10 : ℚ) ^ m) ≤ (2 : ℚ) ^ (l + 1) := by
        exact_mod_cast (Nat.lt_pow_succ_log_self (by norm_num) (10 ^ m)).le
      have hk' : k = -(m : ℤ) := by omega
      rw [hk', zpow_neg, zpow_natCast,
        show (-(l : ℤ) - 1) = -((l + 1 : ℕ) : ℤ) from by push_cast; ring,
        zpow_neg, zpow_natCast, inv_le_inv₀ (by positivity) (by positivity)]
      exact hq
  -- Multiplying by 2^(128-pe) turns it into the claimed bound.
  have hx : (2 : ℚ) ^ (127 : ℕ) ≤ 10 ^ k * 2 ^ (128 - power10Exponent k) := by
    have hmul :
        (2 : ℚ) ^ (power10Exponent k - 1) * 2 ^ (128 - power10Exponent k)
          ≤ 10 ^ k * 2 ^ (128 - power10Exponent k) :=
      mul_le_mul_of_nonneg_right hlog (by positivity)
    rw [← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0),
      show power10Exponent k - 1 + (128 - power10Exponent k) = (127 : ℤ) from by
        ring] at hmul
    simpa using hmul
  exact Nat.le_floor (by push_cast; exact hx)

-- Numerator and denominator of the exact scaled power of ten `10^k·2^(128-pe)`,
-- with negative exponents moved to the denominator. Writing it as a ratio of
-- naturals turns the truncation into a single `Nat` division, so the
-- exponent-wise checks below can run in the kernel.
def power10Num (k : ℤ) : ℕ :=
  10 ^ k.toNat * 2 ^ (128 - power10Exponent k).toNat

def power10Den (k : ℤ) : ℕ :=
  10 ^ (-k).toNat * 2 ^ (power10Exponent k - 128).toNat

-- The scaled exact power of ten is exactly the rational `num / den`.
theorem power10_exact_ratio (k : ℤ) :
    (10 : ℚ) ^ k * 2 ^ (128 - power10Exponent k)
      = (power10Num k : ℚ) / (power10Den k : ℚ) := by
  set pe := power10Exponent k
  have hden : (power10Den k : ℚ) ≠ 0 := by
    rw [power10Den]; positivity
  -- Each pair of exponents in the ratio adds up to the truncated one.
  have hk : k + ((-k).toNat : ℤ) = (k.toNat : ℤ) := by omega
  have hpe : 128 - pe + ((pe - 128).toNat : ℤ) = ((128 - pe).toNat : ℤ) := by
    omega
  rw [eq_div_iff hden, power10Num, power10Den]
  push_cast
  rw [← zpow_natCast (10 : ℚ) (-k).toNat,
    ← zpow_natCast (2 : ℚ) (pe - 128).toNat,
    ← zpow_natCast (10 : ℚ) k.toNat, ← zpow_natCast (2 : ℚ) (128 - pe).toNat,
    show (10 : ℚ) ^ k * 2 ^ (128 - pe) *
        (10 ^ ((-k).toNat : ℤ) * 2 ^ ((pe - 128).toNat : ℤ))
      = (10 ^ k * 10 ^ ((-k).toNat : ℤ)) *
        (2 ^ (128 - pe) * 2 ^ ((pe - 128).toNat : ℤ)) from by ring,
    ← zpow_add₀ (by norm_num : (10 : ℚ) ≠ 0),
    ← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0), hk, hpe]

-- The power-of-ten significand fits in 128 bits: 10^k < 2^pe by construction.
theorem power10_significand_lt (k : ℤ) :
    power10Significand k < 2 ^ 128 := by
  have hlog : (10 : ℚ) ^ k < 2 ^ power10Exponent k := by
    unfold power10Exponent
    split_ifs with hk
    · have hnat : 10 ^ k.toNat < 2 ^ (Nat.log 2 (10 ^ k.toNat) + 1) :=
        Nat.lt_pow_succ_log_self (by norm_num) _
      have hq : ((10 : ℚ) ^ k.toNat)
          < (2 : ℚ) ^ (Nat.log 2 (10 ^ k.toNat) + 1) := by
        exact_mod_cast hnat
      rw [show ((Nat.log 2 (10 ^ k.toNat) : ℤ) + 1)
            = ((Nat.log 2 (10 ^ k.toNat) + 1 : ℕ) : ℤ) from by push_cast; ring,
        zpow_natCast]
      simpa [← zpow_natCast (10 : ℚ) k.toNat, Int.toNat_of_nonneg hk] using hq
    · set m := (-k).toNat with hm
      set l := Nat.log 2 (10 ^ m)
      -- 2^l ≤ 10^m is the defining bound; equality would put a five in 2^l.
      have hne : 2 ^ l ≠ 10 ^ m := by
        intro hcon
        have h5 : (5 : ℕ) ∣ 2 ^ l := by
          rw [hcon]
          exact dvd_pow (⟨2, rfl⟩ : (5 : ℕ) ∣ 10) (by omega)
        have := (Nat.prime_five.dvd_of_dvd_pow h5)
        omega
      have hlt : (2 : ℕ) ^ l < 10 ^ m :=
        lt_of_le_of_ne (Nat.pow_log_le_self 2 (by positivity)) hne
      have hq : ((2 : ℚ) ^ l) < (10 : ℚ) ^ m := by exact_mod_cast hlt
      have hk' : k = -(m : ℤ) := by omega
      rw [hk', zpow_neg, zpow_natCast, zpow_neg, zpow_natCast,
        inv_lt_inv₀ (by positivity) (by positivity)]
      exact hq
  have hx : (10 : ℚ) ^ k * 2 ^ (128 - power10Exponent k) < 2 ^ (128 : ℕ) := by
    have hmul :
        (10 : ℚ) ^ k * 2 ^ (128 - power10Exponent k)
          < 2 ^ power10Exponent k * 2 ^ (128 - power10Exponent k) :=
      mul_lt_mul_of_pos_right hlog (by positivity)
    rw [← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0),
      show power10Exponent k + (128 - power10Exponent k) = (128 : ℤ) from by
        ring] at hmul
    simpa using hmul
  exact (Nat.floor_lt (by positivity)).mpr (by push_cast; exact hx)

/-! ### Exponent alignment and the scale margin -/

-- The shift used by yy's regular path is less than 4.
theorem decimal_shift_lt_four (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    decimalShift e < 4 := by
  unfold decimalShift decimalExponent
  omega

section

-- The finite exponent checks here, and the bounds derived from them, compute
-- powers as large as `10^324` and `2^1074`.
set_option exponentiation.threshold 5000

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

-- Exponent alignment gives 2^(h+1)·2^(128-pe) = 2^(e+128).
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

-- For `e ≠ 0`, `2^e / 10^k` is at least `1 + 2⁻⁶²`.
theorem margin_lower (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) (he0 : e ≠ 0) :
    (1 : ℚ) + 1 / 2 ^ 62 ≤ 2 ^ e * (10 ^ decimalExponent e)⁻¹ := by
  have hb : 0 ≤ e + 1074 := by omega
  have hmz := margin_all e (by simpa [Finset.mem_Icc] using he) he0
  simp only [marginHolds, decide_eq_true_eq] at hmz
  set k := decimalExponent e
  have hk : 0 ≤ k + 324 := by
    show 0 ≤ decimalExponent e + 324
    unfold decimalExponent
    omega
  -- Cast the finite certificate to ℚ and expose the common factor.
  have hcert :
      (2 : ℚ) ^ (1074 : ℕ) * ((2 ^ 62 + 1) * 10 ^ (k + 324).toNat)
        ≤ 2 ^ (62 : ℕ) * (2 ^ (e + 1074).toNat * 10 ^ (324 : ℕ)) := by
    exact_mod_cast hmz
  simp only [← zpow_natCast, Int.toNat_of_nonneg hk, Int.toNat_of_nonneg hb,
    Nat.cast_ofNat] at hcert
  rw [zpow_add₀ (by norm_num : (10 : ℚ) ≠ 0) k 324,
    zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0) e 1074] at hcert
  have hpos : (0 : ℚ) < 2 ^ (1074 : ℤ) * 10 ^ (324 : ℤ) := by positivity
  have hbound : ((2 : ℚ) ^ 62 + 1) * 10 ^ k ≤ 2 ^ 62 * 2 ^ e := by
    have hscaled :
        (2 ^ (1074 : ℤ) * 10 ^ (324 : ℤ)) * (((2 : ℚ) ^ (62 : ℤ) + 1) * 10 ^ k)
          ≤ (2 ^ (1074 : ℤ) * 10 ^ (324 : ℤ)) *
              ((2 : ℚ) ^ (62 : ℤ) * 2 ^ e) := by
      simpa only [mul_assoc, mul_comm, mul_left_comm] using hcert
    exact le_of_mul_le_mul_left hscaled hpos
  -- Normalize the bound by `2^62·10^k`.
  have hp : (0 : ℚ) < 10 ^ k := by positivity
  rw [← div_eq_mul_inv, le_div_iff₀ hp]
  calc
    (1 + 1 / 2 ^ 62 : ℚ) * 10 ^ k
        = ((2 ^ 62 + 1) * 10 ^ k) / 2 ^ 62 := by field_simp
    _ ≤ (2 ^ 62 * 2 ^ e) / 2 ^ 62 := by gcongr
    _ = 2 ^ e := by field_simp

end

/-! ### The longer candidate -/

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
    simpa [h] using decimal_shift_lt_four e hr.2.2
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
theorem dec_one_error_bound
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

/-! ### The packed trim window

yy's `roundD0` and `roundU0` compare the packed value `c` (the last decimal
digit of the integral part, followed by the top 60 bits of `sigLo`) with
`halfUlp`. Both sides are truncated at the same window unit `U`. With

    h = decimalShift e,   p10 = trimSig e,
    N = 10·2^(128-h),     U = 2^(68-h),   W = 2·f·p10 % N

one has `c = W / U`, `halfUlp = p10 / U`, `10·2^60 = N / U`, and
`ten·2^(128-h) + W = 2·f·p10`. Thus the two tests use only the pair
`(W, p10)` measured in units of `U`, and the boundary analysis becomes exact
integer arithmetic rather than an error estimate.
-/

-- The exact power of ten `10^(-k)·2^(128-pe)` as the fraction `num/den`, and
-- its 128-bit truncation `p10`, for the decimal exponent associated with `e`.
-- The trim layer works entirely with these three naturals.
def trimNum (e : ℤ) : ℕ := power10Num (-decimalExponent e)

def trimDen (e : ℤ) : ℕ := power10Den (-decimalExponent e)

def trimSig (e : ℤ) : ℕ := power10Significand (-decimalExponent e)

theorem trim_den_pos (e : ℤ) : 0 < trimDen e := by
  rw [trimDen, power10Den]; positivity

-- The truncation is the floor of the fraction: `p10 = num / den`, so the whole
-- trim layer is natural-number division.
theorem trim_sig_nat (e : ℤ) : trimSig e = trimNum e / trimDen e := by
  rw [trimSig, trimNum, trimDen, power10Significand, power10_exact_ratio]
  exact Nat.floor_div_eq_div _ _

-- Modulus of the packed comparison: the window wraps every 10·2^(128-h).
def trimModulus (e : ℤ) : ℕ := 10 * 2 ^ (128 - decimalShift e)

-- One unit in the last place of the packed comparison.
def trimUnit (e : ℤ) : ℕ := 2 ^ (68 - decimalShift e)

-- The exact remainder above the multiple-of-ten candidate, in units of
-- `2^(h-128)` of the scaled value: `ten·2^(128-h) + trimResidue = 2·f·p10`.
def trimResidue (f : ℕ) (e : ℤ) : ℕ := 2 * f * trimSig e % trimModulus e

-- `scaledSignificand` is the shifted product with the low 64 bits discarded.
theorem scaled_significand_eq (f : ℕ) (e : ℤ) :
    scaledSignificand f e =
      2 ^ decimalShift e * (2 * f * trimSig e) / 2 ^ 64 := by
  show f * 2 ^ (decimalShift e + 1) * trimSig e / 2 ^ 64 = _
  congr 1
  rw [pow_succ]
  ring

-- `sigHi` is the top 64 bits of the shifted 192-bit product.
theorem sig_hi_eq (f : ℕ) (e : ℤ) :
    sigHi f e = 2 ^ decimalShift e * (2 * f * trimSig e) / 2 ^ 128 := by
  show scaledSignificand f e / 2 ^ 64 = _
  rw [scaled_significand_eq, Nat.div_div_eq_div_mul, ← pow_add]

-- `sigLo` is bits 64–127 of the shifted 192-bit product.
theorem sig_lo_eq (f : ℕ) (e : ℤ) :
    sigLo f e
      = 2 ^ decimalShift e * (2 * f * trimSig e) % 2 ^ 128 / 2 ^ 64 := by
  show scaledSignificand f e % 2 ^ 64 = _
  rw [scaled_significand_eq, ← Nat.mod_mul_right_div_self, ← pow_add]

-- 2^128 splits into the shift and the window modulus.
theorem pow_split (e : ℤ) (hsh : decimalShift e < 4) :
    (2 : ℕ) ^ 128 * 10 = 2 ^ decimalShift e * trimModulus e := by
  have h128 :
      (2 : ℕ) ^ 128 = 2 ^ decimalShift e * 2 ^ (128 - decimalShift e) := by
    rw [← pow_add]
    congr 1
    omega
  rw [trimModulus, h128]
  ring

-- Splitting a value at bit 128 and then discarding the low 68 bits is the same
-- as discarding them directly; this is what packs the last digit into `c`.
theorem div_window (r : ℕ) :
    r / 2 ^ 128 * 2 ^ 60 + r % 2 ^ 128 / 2 ^ 68 = r / 2 ^ 68 := by
  conv_rhs => rw [← Nat.div_add_mod r (2 ^ 128)]
  rw [show (2 : ℕ) ^ 128 = 2 ^ 68 * 2 ^ 60 from by norm_num, mul_assoc,
    Nat.mul_add_div (by positivity)]
  ring

-- `halfUlp` is the power-of-ten significand truncated to the window unit.
theorem trim_half_ulp_eq (e : ℤ) (hsh : decimalShift e < 4) :
    trimSig e / 2 ^ 64 / 2 ^ (4 - decimalShift e)
      = trimSig e / trimUnit e := by
  rw [Nat.div_div_eq_div_mul, ← pow_add, trimUnit,
    show 64 + (4 - decimalShift e) = 68 - decimalShift e from by omega]

-- `c` is the window residue truncated to the window unit.
theorem trim_c_eq (f : ℕ) (e : ℤ) (hsh : decimalShift e < 4) :
    sigHi f e % 10 * 2 ^ 60 + sigLo f e / 2 ^ 4
      = trimResidue f e / trimUnit e := by
  set h := decimalShift e with hh
  set z := 2 * f * trimSig e
  set r := 2 ^ h * z % (2 ^ 128 * 10) with hr
  have hpos : 0 < (2 : ℕ) ^ h := by positivity
  have h68 : (2 : ℕ) ^ 68 = 2 ^ h * 2 ^ (68 - h) := by
    rw [← pow_add]
    congr 1
    omega
  have hresidue : r = 2 ^ h * trimResidue f e := by
    rw [hr, pow_split e hsh, Nat.mul_mod_mul_left]
    rfl
  have hscaled : trimResidue f e / trimUnit e = r / 2 ^ 68 := by
    rw [hresidue, h68, trimUnit, ← hh, Nat.mul_div_mul_left _ _ hpos]
  have hhi : sigHi f e % 10 = r / 2 ^ 128 := by
    rw [sig_hi_eq, hr, Nat.mod_mul_right_div_self]
  have hlo : sigLo f e / 2 ^ 4 = r % 2 ^ 128 / 2 ^ 68 := by
    have hmod : 2 ^ h * z % 2 ^ 128 = r % 2 ^ 128 := by
      rw [hr, Nat.mod_mod_of_dvd _ ⟨10, rfl⟩]
    rw [sig_lo_eq, hmod, Nat.div_div_eq_div_mul, ← pow_add]
  rw [hhi, hlo, hscaled, div_window]

-- The multiple-of-ten candidate together with the window residue recovers the
-- full product: `ten·2^(128-h) + W = 2·f·p10`.
theorem trim_residue_add_ten (f : ℕ) (e : ℤ) (hsh : decimalShift e < 4) :
    (sigHi f e - sigHi f e % 10) * 2 ^ (128 - decimalShift e) + trimResidue f e
      = 2 * f * trimSig e := by
  set h := decimalShift e
  set z := 2 * f * trimSig e
  have hpos : 0 < (2 : ℕ) ^ h := by positivity
  have hdiv : sigHi f e / 10 = z / trimModulus e := by
    rw [sig_hi_eq, Nat.div_div_eq_div_mul, pow_split e hsh,
      Nat.mul_div_mul_left _ _ hpos]
  have hten : sigHi f e - sigHi f e % 10 = 10 * (z / trimModulus e) := by
    have hmod := Nat.div_add_mod (sigHi f e) 10
    rw [← hdiv]
    omega
  rw [hten, trimResidue,
    show 10 * (z / trimModulus e) * 2 ^ (128 - h) =
      trimModulus e * (z / trimModulus e) from by rw [trimModulus]; ring]
  exact Nat.div_add_mod z (trimModulus e)

/-! ### What the rounding certificates say about the window

`roundD0` compares `W / U` with `p10 / U`, while `roundU0` compares their sum
with `N / U`. Thus each certificate constrains `W` only up to one window unit
`U`; `dec_ten_down` and `dec_ten_up` read their branches off directly.
Truncation is one-sided, which makes the two directions asymmetric: a
`roundU0` firing on the plain test is always safe, while the one-LSB-offset
test `t1 + 1 = t0` and the trim-down tests can fire one unit early.
-/

-- The window modulus is a whole number of window units.
theorem trim_modulus_eq (e : ℤ) (hsh : decimalShift e < 4) :
    trimModulus e = trimUnit e * (10 * 2 ^ 60) := by
  rw [trimModulus, trimUnit,
    show 128 - decimalShift e = (68 - decimalShift e) + 60 from by omega,
    pow_add]
  ring

/-! ### The separation facts

After the reduction above, the two multiple-of-ten candidates are within half a
ULP exactly when

    trim down:  W + (2f-1)·p10Exact ≤ 2·f·p10,
    trim up:    N + 2·f·p10 ≤ W + (2f+1)·p10Exact,

writing `p10Exact` for the exact power of ten `10^(-k)·2^(128-pe)`, which
satisfies `p10 ≤ p10Exact < p10 + 1`. Substituting the window identity
`ten·2^(128-h) + W = 2·f·p10` and `p10Exact·s = 2^(128-h)`, and writing
`ten = 10·M`, both read as comparisons of the exact binary rounding boundary
with the decimal grid the trim targets:

    trim down:  2^(e-1)·(2f-1) ≤ M·10^(k+1),
    trim up:    (M+1)·10^(k+1) ≤ 2^(e-1)·(2f+1).

So the quantity to control is `Λ = M·10^(k+1) - 2^(e-1)·(2f∓1)`. Exact ties are
the case `Λ = 0`: since `2f∓1` is odd it carries no twos, so the boundary can
only land on `10^(k+1)·ℤ` when `e - 1 ≥ k + 1` supplies them, and the remaining
power of two is a unit modulo `5^(k+1)`, leaving the single residue class
`2f∓1 ≡ 0 (mod 5^(k+1))`—an arithmetic progression in `f`, not a magnitude
condition. With `2f∓1 < 2^54` this needs `5^(k+1) < 2^54`, so ties exist only
for `k ≤ 22`; there the two bounds hold with equality, which is exactly why the
parity test turns them into `≤` for even `f` and `<` for odd `f`. What must be
ruled out is a near miss, `Λ ≠ 0` of the wrong sign.

A magnitude bound cannot do that on its own. For `k ≥ 0` and `e ≥ k + 2`, both
terms of `Λ` are divisible by `2^(k+1)`, so the scaled defect is a multiple of
`2^(129-h)/5^k` while `trim_low_bits` only bounds it by `2^69`; that forces
`Λ = 0` only while `5^k ≤ 2^61`, i.e. `k ≤ 26`. Beyond that the quantum is
smaller than the uncertainty and the separation becomes genuinely arithmetic,
so the development below clears denominators instead. Writing `num/den` for the
exact power of ten and `τ = num % den`, the two bounds become

    trim down:  trimGap ≤ num,
    trim up:    trimScale ≤ trimGap + num,

and `trim_gap_mod` identifies `trimGap` with `2·num·f mod trimScale` as long as
it has not wrapped, which the packed comparisons guarantee. Thus a violation
would put a single modular progression inside an interval of width below
`den·U`, a window of relative width `U/N ≈ 2^(-63.3)`. The finite certificates
rule this out for every exponent. This is where verify.py counts solutions with
`floor_sum`; here a refutation certificate reaches the same conclusion with one
small check per window, and the exponent range is kernel-checked in blocks of
64.
-/

-- The exact distance from the multiple-of-ten candidate to the scaled value,
-- with the denominator of the exact power of ten cleared: `den·W + 2·f·τ`.
def trimGap (f : ℕ) (e : ℤ) : ℕ :=
  trimDen e * trimResidue f e + 2 * f * (trimNum e % trimDen e)

-- The window modulus with the same denominator cleared.
def trimScale (e : ℤ) : ℕ := trimModulus e * trimDen e

-- How far a gap can be from the multiple-of-ten candidate and still be accepted
-- by a packed comparison: `den·(p10 + U)`. Written as `num / den` rather than
-- `trimSig` so the certificate remains purely natural-number computation.
def trimBnd (e : ℤ) : ℕ := trimDen e * (trimNum e / trimDen e + trimUnit e)

-- Integer form of `2^54·(p10Exact - p10) ≤ p10 % U`, with the denominator
-- cleared.
def trimLowBitsHolds (e : ℤ) : Bool :=
  let num := trimNum e
  let den := trimDen e
  decide (2 ^ 54 * (num % den) ≤ num / den % trimUnit e * den)

-- An exact trim-up tie forces `N ∣ (2f+1)·p10`; since `2f+1` is odd and
-- `N = 5·2^(129-h)`, all of that power of two must come from `p10`.
def trimTieNeedsKZero (e : ℤ) : Bool :=
  let num := trimNum e
  let den := trimDen e
  decide (decimalExponent e = 0 ∨ ¬2 ^ (129 - decimalShift e) ∣ num / den)

section
set_option exponentiation.threshold 5000

theorem trim_low_bits_all :
    ∀ e ∈ Finset.Icc (-1074 : ℤ) 971, trimLowBitsHolds e = true := by decide

theorem trim_tie_needs_k_zero_all :
    ∀ e ∈ Finset.Icc (-1074 : ℤ) 971, trimTieNeedsKZero e = true := by decide

end

-- The low bits `p10 % U` discarded by the packed comparison dominate the
-- truncation error `p10Exact - p10 = τ/den`. This settles every case where the
-- packed comparison is strict: the margin is wide, but which bits of the power
-- of ten survive truncation is not a magnitude property, so it is checked per
-- exponent.
theorem trim_low_bits (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    2 ^ 54 * (trimNum e % trimDen e)
      ≤ trimSig e % trimUnit e * trimDen e := by
  have hcert := trim_low_bits_all e (by simpa [Finset.mem_Icc] using he)
  simp only [trimLowBitsHolds, decide_eq_true_eq] at hcert
  rwa [trim_sig_nat]

-- An exact trim-up tie is only possible for `k = 0`. This is why `roundU0`
-- needs its `k = 0 ∧ t1 = t0` branch: at `k = 0`, `p10` is exactly `2^127`,
-- so it is untruncated and divisible by the window unit, and a genuine tie
-- appears as `t1 = t0` rather than `t1 + 1 = t0`.
theorem trim_tie_k_zero (f : ℕ) (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971)
    (hsh : decimalShift e < 4)
    (hdvd : trimModulus e ∣ (2 * f + 1) * trimSig e) :
    decimalExponent e = 0 := by
  have hcert := trim_tie_needs_k_zero_all e (by simpa [Finset.mem_Icc] using he)
  simp only [trimTieNeedsKZero, decide_eq_true_eq, ← trim_sig_nat] at hcert
  refine hcert.resolve_right (not_not.mpr ?_)
  refine Nat.Coprime.dvd_of_dvd_mul_left
    (Nat.Coprime.pow_left _
      ((Nat.prime_two.coprime_iff_not_dvd).mpr (by omega)))
    (dvd_trans ⟨5, ?_⟩ hdvd)
  rw [trimModulus,
    show 129 - decimalShift e = (128 - decimalShift e) + 1 from by omega,
    pow_succ]
  ring

/-! ### Refuting the trim windows

Both bounds reduce to one arithmetic question per exponent. Writing `num/den`
for the exact power of ten, `modulus = N·den` and `g = 2·num`, a violation
forces `g·f mod modulus` into an interval of width below `den·U`. Such a window
is refuted by a single multiplier `q`: writing `y = g·f - modulus·j` for the
residue and `r = g·q - modulus·p` for the approximation error of
`p/q ≈ g/modulus`,

    modulus·(p·f - q·j) = q·y - f·r,

so if `q·y - f·r` stays strictly between two consecutive multiples of `modulus`
throughout the box `f ∈ [F₀,F₁]`, `y ∈ [lo,hi]`, no `f` can hit the window.
Taking `q` to be a continued-fraction denominator of `g/modulus` makes the
range about `2·√(n·(hi-lo)/modulus) ≈ 2⁻⁵` wide, where `n = 2^52` is the
number of significands in the box, so it fits between multiples with room to
spare. The multiplier is a witness, not an assumption: only the check needs a
proof.
-/

def modWindowRefuted (g modulus f0 f1 lo hi q : ℤ) : Bool :=
  let p := (2 * (g * q) + modulus) / (2 * modulus)
  let r := g * q - modulus * p
  -- `f · r` runs between the two endpoint values, in whichever order the sign
  -- of `r` dictates.
  let lo' := q * lo - max (f0 * r) (f1 * r)
  let hi' := q * hi - min (f0 * r) (f1 * r)
  decide (0 < q ∧ modulus * (lo' / modulus) < lo'
    ∧ hi' < modulus * (lo' / modulus) + modulus)

-- A value squeezed strictly between consecutive multiples of `modulus` is
-- absurd.
theorem window_gap_absurd {modulus lo' hi' v : ℤ} (hmodulus : 0 < modulus)
    (hlo : lo' ≤ modulus * v) (hhi : modulus * v ≤ hi')
    (hgap_lo : modulus * (lo' / modulus) < lo')
    (hgap_hi : hi' < modulus * (lo' / modulus) + modulus) :
    False := by
  have hv_lo : lo' / modulus < v :=
    lt_of_mul_lt_mul_left (lt_of_lt_of_le hgap_lo hlo) hmodulus.le
  have hv_hi : v < lo' / modulus + 1 := by
    refine lt_of_mul_lt_mul_left (a := modulus) ?_ hmodulus.le
    calc
      modulus * v ≤ hi' := hhi
      _ < modulus * (lo' / modulus) + modulus := hgap_hi
      _ = modulus * (lo' / modulus + 1) := by ring
  omega

-- The certificate identity and the resulting bounds over the significand box.
theorem window_bounds {g modulus f0 f1 lo hi q p r f j y : ℤ}
    (hq : 0 < q) (hr : r = g * q - modulus * p)
    (hf0 : f0 ≤ f) (hf1 : f ≤ f1)
    (hy : y = g * f - modulus * j) (hlo : lo ≤ y) (hhi : y ≤ hi) :
    q * lo - max (f0 * r) (f1 * r) ≤ modulus * (p * f - q * j) ∧
      modulus * (p * f - q * j) ≤ q * hi - min (f0 * r) (f1 * r) := by
  have hkey : modulus * (p * f - q * j) = q * y - f * r := by
    rw [hy, hr]
    ring
  have hqy0 : q * lo ≤ q * y := mul_le_mul_of_nonneg_left hlo hq.le
  have hqy1 : q * y ≤ q * hi := mul_le_mul_of_nonneg_left hhi hq.le
  have hfr : min (f0 * r) (f1 * r) ≤ f * r ∧ f * r ≤ max (f0 * r) (f1 * r) := by
    rcases le_total 0 r with hr0 | hr0
    · exact ⟨le_trans (min_le_left _ _) (mul_le_mul_of_nonneg_right hf0 hr0),
        le_trans (mul_le_mul_of_nonneg_right hf1 hr0) (le_max_right _ _)⟩
    · exact ⟨le_trans (min_le_right _ _) (mul_le_mul_of_nonpos_right hf1 hr0),
        le_trans (mul_le_mul_of_nonpos_right hf0 hr0) (le_max_left _ _)⟩
  exact ⟨by linarith [hfr.2], by linarith [hfr.1]⟩

theorem not_window_hit {g modulus f0 f1 lo hi q f j y : ℤ}
    (hmodulus : 0 < modulus)
    (hcert : modWindowRefuted g modulus f0 f1 lo hi q = true)
    (hf0 : f0 ≤ f) (hf1 : f ≤ f1)
    (hy : y = g * f - modulus * j)
    (hlo : lo ≤ y) (hhi : y ≤ hi) :
    False := by
  simp only [modWindowRefuted, decide_eq_true_eq] at hcert
  obtain ⟨hq, hgap0, hgap1⟩ := hcert
  let p := (2 * (g * q) + modulus) / (2 * modulus)
  obtain ⟨hb0, hb1⟩ :=
    window_bounds
      (p := p)
      (r := g * q - modulus * p)
      hq rfl hf0 hf1 hy hlo hhi
  exact window_gap_absurd hmodulus hb0 hb1 hgap0 hgap1

-- The two windows a violation would have to hit: `g·f mod modulus` just above
-- `num` for the trim-down bound, or just below `modulus - num` for the trim-up
-- bound. Both are passed in already computed, so that the search below can
-- retry a window without recomputing the power of ten.
def modWindowsRefuted (g modulus lo1 hi1 lo2 hi2 q : ℤ) : Bool :=
  modWindowRefuted g modulus (2 ^ 52 + 1) (2 ^ 53 - 1) lo1 hi1 q &&
    modWindowRefuted g modulus (2 ^ 52 + 1) (2 ^ 53 - 1) lo2 hi2 q

-- The multiplier is searched for in the kernel rather than tabulated.
-- Convergent denominators of `g/modulus` give the relevant best rational
-- approximations. The Euclidean remainder `v` that produces `qc` is the error
-- `|g·qc - modulus·p|`; it must fall below `modulus/2^53` before a window can
-- be refuted. Testing that first skips the small denominators and leaves at
-- most three window checks per exponent.
def modCertSearch (g modulus lo1 hi1 lo2 hi2 : ℤ) : ℕ → ℤ → ℤ → ℤ → ℤ → ℤ
  | 0, _, _, _, qc => qc
  | n + 1, u, v, qp, qc =>
    if decide (2 ^ 53 * v < modulus)
        && modWindowsRefuted g modulus lo1 hi1 lo2 hi2 qc then
      qc
    else if v = 0 then
      qc
    else
      modCertSearch g modulus lo1 hi1 lo2 hi2 n v (u % v) qc (u / v * qc + qp)

def trimWindowsEmpty (e : ℤ) : Bool :=
  let num : ℤ := trimNum e
  let bnd : ℤ := trimBnd e
  let modulus : ℤ := trimScale e
  modWindowsRefuted (2 * num) modulus
    (num + 1) (bnd - 1) (modulus - bnd) (modulus - num - 1)
    (modCertSearch (2 * num) modulus
      (num + 1) (bnd - 1) (modulus - bnd) (modulus - num - 1)
      80 modulus (2 * num % modulus) 0 1)

/-! #### Certifying the exponent range

The 2046 exponents are certified in blocks of 64. A `decide` keeps every
intermediate of the searches it runs alive until the whole declaration is
checked, and the ladder produces a few hundred numbers of about a thousand bits
per exponent, so the cost of one block grows quadratically in its width. The
full range in a single declaration needs tens of gigabytes, while 32 separate
declarations of 64 exponents stay under a second each.
-/

def trimChunk (i : ℕ) : Bool :=
  (List.range 64).all fun j => trimWindowsEmpty (-1074 + 64 * i + j)

section

set_option exponentiation.threshold 5000
theorem trim_chunk_00 : trimChunk 0 = true := by decide
theorem trim_chunk_01 : trimChunk 1 = true := by decide
theorem trim_chunk_02 : trimChunk 2 = true := by decide
theorem trim_chunk_03 : trimChunk 3 = true := by decide
theorem trim_chunk_04 : trimChunk 4 = true := by decide
theorem trim_chunk_05 : trimChunk 5 = true := by decide
theorem trim_chunk_06 : trimChunk 6 = true := by decide
theorem trim_chunk_07 : trimChunk 7 = true := by decide
theorem trim_chunk_08 : trimChunk 8 = true := by decide
theorem trim_chunk_09 : trimChunk 9 = true := by decide
theorem trim_chunk_10 : trimChunk 10 = true := by decide
theorem trim_chunk_11 : trimChunk 11 = true := by decide
theorem trim_chunk_12 : trimChunk 12 = true := by decide
theorem trim_chunk_13 : trimChunk 13 = true := by decide
theorem trim_chunk_14 : trimChunk 14 = true := by decide
theorem trim_chunk_15 : trimChunk 15 = true := by decide
theorem trim_chunk_16 : trimChunk 16 = true := by decide
theorem trim_chunk_17 : trimChunk 17 = true := by decide
theorem trim_chunk_18 : trimChunk 18 = true := by decide
theorem trim_chunk_19 : trimChunk 19 = true := by decide
theorem trim_chunk_20 : trimChunk 20 = true := by decide
theorem trim_chunk_21 : trimChunk 21 = true := by decide
theorem trim_chunk_22 : trimChunk 22 = true := by decide
theorem trim_chunk_23 : trimChunk 23 = true := by decide
theorem trim_chunk_24 : trimChunk 24 = true := by decide
theorem trim_chunk_25 : trimChunk 25 = true := by decide
theorem trim_chunk_26 : trimChunk 26 = true := by decide
theorem trim_chunk_27 : trimChunk 27 = true := by decide
theorem trim_chunk_28 : trimChunk 28 = true := by decide
theorem trim_chunk_29 : trimChunk 29 = true := by decide
theorem trim_chunk_30 : trimChunk 30 = true := by decide
theorem trim_chunk_31 : trimChunk 31 = true := by decide

end

theorem trim_chunk_all (i : ℕ) (hi : i < 32) : trimChunk i = true := by
  interval_cases i
  exacts [
    trim_chunk_00, trim_chunk_01, trim_chunk_02, trim_chunk_03,
    trim_chunk_04, trim_chunk_05, trim_chunk_06, trim_chunk_07,
    trim_chunk_08, trim_chunk_09, trim_chunk_10, trim_chunk_11,
    trim_chunk_12, trim_chunk_13, trim_chunk_14, trim_chunk_15,
    trim_chunk_16, trim_chunk_17, trim_chunk_18, trim_chunk_19,
    trim_chunk_20, trim_chunk_21, trim_chunk_22, trim_chunk_23,
    trim_chunk_24, trim_chunk_25, trim_chunk_26, trim_chunk_27,
    trim_chunk_28, trim_chunk_29, trim_chunk_30, trim_chunk_31
  ]

theorem trim_windows_empty (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    trimWindowsEmpty e = true := by
  let chunk := (e + 1074) / 64
  let offset := e - 64 * chunk + 1074
  have hc := trim_chunk_all chunk.toNat (by
    dsimp [chunk]
    omega)
  simp only [trimChunk, List.all_eq_true, List.mem_range] at hc
  have hr := hc offset.toNat (by
    dsimp [offset, chunk]
    omega)
  rwa [show -1074 + 64 * (chunk.toNat : ℤ) + (offset.toNat : ℤ) = e from by
    dsimp [offset, chunk]
    omega] at hr

-- Scaling the window residue by `den` and adding back the truncation error
-- `2·f·τ` preserves the residue modulo `n·den`, provided the sum has not
-- wrapped.
theorem trim_mod_shift (p den τ n f : ℕ)
    (hlt : den * (2 * f * p % n) + 2 * f * τ < n * den) :
    2 * (p * den + τ) * f % (n * den)
      = den * (2 * f * p % n) + 2 * f * τ := by
  have hsplit := Nat.div_add_mod (2 * f * p) n
  have hkey : 2 * (p * den + τ) * f
      = n * den * (2 * f * p / n)
        + (den * (2 * f * p % n) + 2 * f * τ) := by
    calc 2 * (p * den + τ) * f = den * (2 * f * p) + 2 * f * τ := by ring
      _ = den * (n * (2 * f * p / n) + 2 * f * p % n) + 2 * f * τ := by
          rw [hsplit]
      _ = n * den * (2 * f * p / n)
            + (den * (2 * f * p % n) + 2 * f * τ) := by ring
  rw [hkey, Nat.mul_add_mod, Nat.mod_eq_of_lt hlt]

theorem trim_gap_mod (f : ℕ) (e : ℤ) (hlt : trimGap f e < trimScale e) :
    2 * trimNum e * f % trimScale e = trimGap f e := by
  rw [trimGap, trimResidue, trim_sig_nat, trimScale] at hlt ⊢
  rw [show 2 * trimNum e * f
      = 2 * (trimNum e / trimDen e * trimDen e + trimNum e % trimDen e) * f
      from by rw [Nat.div_add_mod']]
  exact trim_mod_shift _ _ _ _ _ hlt

-- A gap landing in a refuted window is impossible.
theorem trim_no_window_hit {lo hi q : ℤ} (f : ℕ) (e : ℤ)
    (h : Regular f e)
    (hcert : modWindowRefuted (2 * (trimNum e : ℤ)) (trimScale e : ℤ)
      (2 ^ 52 + 1) (2 ^ 53 - 1) lo hi q = true)
    (hwrap : trimGap f e < trimScale e)
    (hlo : lo ≤ (trimGap f e : ℤ))
    (hhi : (trimGap f e : ℤ) ≤ hi) :
    False := by
  obtain ⟨hf_lo, hf_hi, _, _⟩ := h
  have hscale_pos : (0 : ℤ) < (trimScale e : ℤ) :=
    Int.natCast_pos.mpr (by
      rw [trimScale, trimModulus]
      exact Nat.mul_pos (by positivity) (trim_den_pos e))
  refine not_window_hit (f := (f : ℤ))
    (j := ((2 * trimNum e * f / trimScale e : ℕ) : ℤ))
    hscale_pos hcert (by omega) (by omega) ?_ hlo hhi
  have hsplit := Nat.div_add_mod (2 * trimNum e * f) (trimScale e)
  rw [trim_gap_mod f e hwrap] at hsplit
  have hz :
      ((trimScale e * (2 * trimNum e * f / trimScale e) + trimGap f e : ℕ) : ℤ)
        = ((2 * trimNum e * f : ℕ) : ℤ) := by
    exact_mod_cast hsplit
  push_cast at hz ⊢
  linarith

-- `p10 + U` is far below the window modulus, so a gap bounded by
-- `den·(p10 + U)` has not wrapped.
theorem trim_bnd_le_scale (e : ℤ) (hsh : decimalShift e < 4) :
    trimBnd e ≤ trimScale e := by
  rw [trimBnd, ← trim_sig_nat]
  have hu_le : trimUnit e ≤ 2 ^ 68 := by
    rw [trimUnit]; exact Nat.pow_le_pow_right (by norm_num) (by omega)
  have hn_ge : 10 * 2 ^ 125 ≤ trimModulus e := by
    rw [trimModulus]
    exact Nat.mul_le_mul_left _ (Nat.pow_le_pow_right (by norm_num) (by omega))
  have hp_lt : trimSig e < 2 ^ 128 := power10_significand_lt _
  have hgap : (2 : ℕ) ^ 128 + 2 ^ 68 ≤ 10 * 2 ^ 125 := by norm_num
  have hwindow : trimSig e + trimUnit e ≤ trimModulus e := by omega
  rw [trimScale, Nat.mul_comm (trimModulus e)]
  exact Nat.mul_le_mul_left _ hwindow

-- `p10Exact ≥ 2^127`, with the denominator cleared.
theorem trim_num_lower (e : ℤ) : 2 ^ 127 * trimDen e ≤ trimNum e := by
  have h : 2 ^ 127 ≤ trimNum e / trimDen e := by
    rw [← trim_sig_nat]; exact power10_significand_lower _
  exact (Nat.le_div_iff_mul_le (trim_den_pos e)).mp h

/-! ### From window counts to trim bounds

Truncating to blocks of `U` is what the packed comparisons do, so each trim
bound has to turn a comparison of block counts back into one of the untruncated
values. The first two facts below do that, non-strictly and strictly, since the
discarded low bits are worth less than one block; the third says that truncating
a sum never overshoots it. All three are stated for a generic block size `u` and
always instantiated with `U`.
-/

theorem add_mod_lt_of_div_le {a b u : ℕ} (hu : 0 < u)
    (h : a / u ≤ b / u) : a + b % u < b + u := by
  have ha := Nat.div_add_mod a u
  have hb := Nat.div_add_mod b u
  have hmod : a % u < u := Nat.mod_lt _ hu
  have hscaled : u * (a / u) ≤ u * (b / u) := Nat.mul_le_mul_left _ h
  omega

theorem add_mod_succ_le_of_div_lt {a b u : ℕ} (hu : 0 < u)
    (h : a / u < b / u) : a + b % u + 1 ≤ b := by
  have ha := Nat.div_add_mod a u
  have hb := Nat.div_add_mod b u
  have hmod : a % u < u := Nat.mod_lt _ hu
  have hscaled : u * (a / u) + u ≤ u * (b / u) := by
    rw [← Nat.mul_succ]; exact Nat.mul_le_mul_left _ h
  omega

theorem mul_div_add_div_le (a b u : ℕ) :
    u * (a / u + b / u) ≤ a + b := by
  have ha := Nat.div_add_mod a u
  have hb := Nat.div_add_mod b u
  calc
    u * (a / u + b / u) = u * (a / u) + u * (b / u) := by ring
    _ ≤ a + b := by omega

theorem trim_unit_pos (e : ℤ) : 0 < trimUnit e := by
  rw [trimUnit]; positivity

-- `den·p10 + τ = num`: the truncated power of ten and the low bits it dropped.
theorem trim_num_split (e : ℤ) :
    trimDen e * trimSig e + trimNum e % trimDen e = trimNum e := by
  rw [trim_sig_nat]; exact Nat.div_add_mod _ _

theorem trim_scale_split (e : ℤ) :
    trimDen e * trimModulus e = trimScale e := by
  rw [trimScale]; ring

-- The bridge every trim bound crosses: the gap is `den` times the residue plus
-- the truncation error `2·f·τ`, and `trim_low_bits` keeps that error inside the
-- low bits of `p10` that the packed comparison discards anyway.
theorem trim_gap_sandwich (f : ℕ) (e : ℤ) (h : Regular f e) :
    trimDen e * trimResidue f e ≤ trimGap f e ∧
      trimGap f e ≤ trimDen e * trimResidue f e
        + trimSig e % trimUnit e * trimDen e := by
  rw [trimGap]
  have htrunc : 2 * f * (trimNum e % trimDen e)
      ≤ trimSig e % trimUnit e * trimDen e := by
    calc
      2 * f * (trimNum e % trimDen e)
          ≤ 2 ^ 54 * (trimNum e % trimDen e) :=
        Nat.mul_le_mul_right _ (by have := h.2.1; omega)
      _ ≤ trimSig e % trimUnit e * trimDen e := trim_low_bits e h.2.2
  omega

-- What the packed comparisons say about the untruncated values: a packed sum
-- of at least `m` window units is worth at least `U·m`.
theorem trim_pack (f : ℕ) (e : ℤ) (m : ℕ)
    (hb : m ≤ trimResidue f e / trimUnit e + trimSig e / trimUnit e) :
    trimUnit e * m ≤ trimResidue f e + trimSig e :=
  le_trans (Nat.mul_le_mul_left _ hb) (mul_div_add_div_le _ _ _)

-- Under the trim-down comparison the gap stays within `den·(p10 + U)`: the low
-- bits of `p10` cover the truncation error, and the comparison leaves at most
-- one window unit of slack.
theorem trim_gap_box (f : ℕ) (e : ℤ) (h : Regular f e)
    (hcmp : trimResidue f e / trimUnit e ≤ trimSig e / trimUnit e) :
    trimGap f e < trimBnd e := by
  rw [trimBnd, ← trim_sig_nat]
  have hslack := add_mod_lt_of_div_le (trim_unit_pos e) hcmp
  have hscaled : trimDen e * trimResidue f e
      + trimSig e % trimUnit e * trimDen e
      < trimDen e * (trimSig e + trimUnit e) := by
    rw [mul_comm (trimSig e % trimUnit e), ← mul_add]
    exact mul_lt_mul_of_pos_left hslack (trim_den_pos e)
  have hsand : trimGap f e
      ≤ trimDen e * trimResidue f e + trimSig e % trimUnit e * trimDen e :=
    (trim_gap_sandwich f e h).2
  exact lt_of_le_of_lt hsand hscaled

-- The certificates' semantic content: the gap is never strictly between `num`
-- and the window edge, nor within the window edge of the modulus while still
-- short of `scale - num`. Below this theorem is modular arithmetic of
-- `2·num·f mod scale`; above it is yy correctness.
theorem trim_gap_not_in_windows (f : ℕ) (e : ℤ) (h : Regular f e) :
    ¬(trimNum e < trimGap f e ∧ trimGap f e < trimBnd e) ∧
      ¬(trimScale e ≤ trimGap f e + trimBnd e ∧
        trimGap f e + trimNum e < trimScale e) := by
  have hcert := trim_windows_empty e h.2.2
  simp only [trimWindowsEmpty, modWindowsRefuted, Bool.and_eq_true] at hcert
  obtain ⟨hlo_cert, hhi_cert⟩ := hcert
  constructor
  · rintro ⟨hlo, hhi⟩
    have hbnd := trim_bnd_le_scale e (decimal_shift_lt_four e h.2.2)
    exact trim_no_window_hit f e h hlo_cert (by omega) (by omega) (by omega)
  · rintro ⟨hlo, hhi⟩
    exact trim_no_window_hit f e h hhi_cert (by omega) (by omega) (by omega)

-- Trim-down soundness: if the packed comparison reads `c ≤ halfUlp`, then
-- the exact gap is at most `num`; otherwise it would lie in a forbidden window.
theorem trim_gap_le (f : ℕ) (e : ℤ) (h : Regular f e)
    (hcmp : trimResidue f e / trimUnit e ≤ trimSig e / trimUnit e) :
    trimGap f e ≤ trimNum e := by
  by_contra! hcon
  exact (trim_gap_not_in_windows f e h).1 ⟨hcon, trim_gap_box f e h hcmp⟩

-- The strict version needs no certificate: one full window unit of slack
-- already exceeds the truncation error of the power of ten.
theorem trim_gap_lt (f : ℕ) (e : ℤ) (h : Regular f e)
    (hcmp : trimResidue f e / trimUnit e < trimSig e / trimUnit e) :
    trimGap f e < trimNum e := by
  -- A whole window unit of slack, scaled by `den`, outweighs the low bits.
  have hscaled : trimDen e * trimResidue f e
      + trimSig e % trimUnit e * trimDen e < trimDen e * trimSig e := by
    have hslack := add_mod_succ_le_of_div_lt (trim_unit_pos e) hcmp
    rw [mul_comm (trimSig e % trimUnit e), ← mul_add]
    exact mul_lt_mul_of_pos_left (by omega) (trim_den_pos e)
  have hnum : trimDen e * trimSig e + trimNum e % trimDen e = trimNum e :=
    trim_num_split e
  have hsand : trimGap f e
      ≤ trimDen e * trimResidue f e + trimSig e % trimUnit e * trimDen e :=
    (trim_gap_sandwich f e h).2
  omega

-- Trim-up soundness: if the packed sum is within one unit of the modulus,
-- `gap + num` reaches the modulus after clearing the denominator.
theorem trim_scale_le (f : ℕ) (e : ℤ) (h : Regular f e)
    (hcmp : trimModulus e ≤ trimResidue f e + trimSig e + trimUnit e) :
    trimScale e ≤ trimGap f e + trimNum e := by
  by_contra! hcon
  -- The comparison keeps the gap within the window edge of the modulus.
  have hfar : trimScale e ≤ trimGap f e + trimBnd e := by
    rw [trimBnd, ← trim_sig_nat]
    have hscaled : trimScale e
        ≤ trimDen e * trimResidue f e
          + trimDen e * (trimSig e + trimUnit e) := by
      rw [← trim_scale_split e, ← mul_add, ← Nat.add_assoc]
      exact Nat.mul_le_mul_left _ hcmp
    have hsand : trimDen e * trimResidue f e ≤ trimGap f e :=
      (trim_gap_sandwich f e h).1
    omega
  exact (trim_gap_not_in_windows f e h).2 ⟨hfar, hcon⟩

-- Strict trim-up soundness for the final `t0 ≤ t1` test. Equality is an exact
-- tie, which forces `k = 0` by `trim_tie_k_zero`; that case belongs to the
-- dedicated `k = 0 ∧ t1 = t0` branch instead. -/
theorem trim_scale_lt (f : ℕ) (e : ℤ) (h : Regular f e)
    (hb : 10 * 2 ^ 60 ≤ trimResidue f e / trimUnit e + trimSig e / trimUnit e)
    (hne : ¬(decimalExponent e = 0 ∧ trimResidue f e / trimUnit e
      + trimSig e / trimUnit e = 10 * 2 ^ 60)) :
    trimScale e < trimGap f e + trimNum e := by
  have hsh : decimalShift e < 4 := decimal_shift_lt_four e h.2.2
  have hu_pos := trim_unit_pos e
  have hmodeq : trimModulus e = trimUnit e * (10 * 2 ^ 60) :=
    trim_modulus_eq e hsh
  have hcmp : trimModulus e ≤ trimResidue f e + trimSig e := by
    rw [hmodeq]; exact trim_pack f e _ hb
  -- Equality would be an exact tie, and only `k = 0` admits one.
  have hlt : trimModulus e < trimResidue f e + trimSig e := by
    rcases lt_or_eq_of_le hcmp with hlt | heq
    · exact hlt
    · exfalso
      have hw : trimResidue f e = 2 * f * trimSig e % trimModulus e := rfl
      have hdvd : trimModulus e ∣ (2 * f + 1) * trimSig e := by
        refine ⟨2 * f * trimSig e / trimModulus e + 1, ?_⟩
        have hq := Nat.div_add_mod (2 * f * trimSig e) (trimModulus e)
        calc (2 * f + 1) * trimSig e = 2 * f * trimSig e + trimSig e := by ring
          _ = trimModulus e * (2 * f * trimSig e / trimModulus e)
              + (trimResidue f e + trimSig e) := by rw [hw]; omega
          _ = trimModulus e * (2 * f * trimSig e / trimModulus e + 1) := by
              rw [← heq]; ring
      refine hne ⟨trim_tie_k_zero f e h.2.2 hsh hdvd, ?_⟩
      -- An exact tie is seen as one by the packed comparison too.
      have h4 : trimUnit e
          * (trimResidue f e / trimUnit e + trimSig e / trimUnit e)
          ≤ trimUnit e * (10 * 2 ^ 60) := by
        rw [← hmodeq, heq]; exact mul_div_add_div_le _ _ _
      exact Nat.le_antisymm (Nat.le_of_mul_le_mul_left h4 hu_pos) hb
  -- Clearing the denominator preserves the strict inequality.
  have hscaled : trimScale e
      < trimDen e * trimResidue f e + trimDen e * trimSig e := by
    rw [← trim_scale_split e, ← mul_add]
    exact mul_lt_mul_of_pos_left hlt (trim_den_pos e)
  have hnum : trimDen e * trimSig e + trimNum e % trimDen e = trimNum e :=
    trim_num_split e
  have hsand : trimDen e * trimResidue f e ≤ trimGap f e :=
    (trim_gap_sandwich f e h).1
  omega

-- The gap can overshoot the modulus, but by less than `num`: the residue is
-- below the modulus and the truncation error contributes at most `2^68·den`,
-- which `num ≥ 2^127·den` absorbs. This is the free side of the trim-up bound.
theorem trim_gap_lt_scale_add (f : ℕ) (e : ℤ) (h : Regular f e) :
    trimGap f e < trimScale e + trimNum e := by
  have hw : trimResidue f e < trimModulus e :=
    Nat.mod_lt _ (by rw [trimModulus]; positivity)
  have hres : trimDen e * trimResidue f e < trimScale e := by
    rw [← trim_scale_split e]
    exact mul_lt_mul_of_pos_left hw (trim_den_pos e)
  have hu : trimSig e % trimUnit e < 2 ^ 68 :=
    lt_of_lt_of_le (Nat.mod_lt _ (trim_unit_pos e))
      (by rw [trimUnit]; exact Nat.pow_le_pow_right (by norm_num) (by omega))
  have htrunc : trimSig e % trimUnit e * trimDen e ≤ trimNum e := by
    calc
      trimSig e % trimUnit e * trimDen e
          ≤ 2 ^ 68 * trimDen e := Nat.mul_le_mul_right _ hu.le
      _ ≤ 2 ^ 127 * trimDen e :=
        Nat.mul_le_mul_right _
          (Nat.pow_le_pow_right (by norm_num) (by norm_num))
      _ ≤ trimNum e := trim_num_lower e
  have hsand : trimGap f e
      ≤ trimDen e * trimResidue f e + trimSig e % trimUnit e * trimDen e :=
    (trim_gap_sandwich f e h).2
  omega

/-! ### From integer bounds to half-ULP bounds

The trimmed-candidate comparisons below reduce to comparisons of naturals in
the scale `trimMul`. Two integer identities translate the trimmed candidates
into this scale: `trim_mul_ten` sends `10` to `trimScale`, and
`trim_mul_dec_ten` sends the multiple-of-ten candidate to
`2·f·trimNum - trimGap`. The power of ten enters only through `trim_mul_eq`,
which expresses `trimMul` as `trimNum` times the inverse scale
`s = 2^(1-e)·10^k`. Thus `|cand - x| ≤ u/2` reduces to
`|cand·trimMul - 2·f·trimNum| ≤ trimNum`.
-/

def trimMul (e : ℤ) : ℕ := 2 ^ (128 - decimalShift e) * trimDen e

-- Exponent alignment: the inverse scale `s = 2^(1-e)·10^k` turns the
-- power-of-ten factor `10^(-k)·2^(128-pe)` into `2^(128-h)`.
theorem exact_scale (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    let k := decimalExponent e
    let pe := power10Exponent (-k)
    (10 : ℚ) ^ (-k) * 2 ^ (128 - pe) * (2 ^ (1 - e) * 10 ^ k)
      = 2 ^ (128 - decimalShift e) := by
  intro k pe
  have h10 : (10 : ℚ) ^ (-k) * 10 ^ k = 1 := by
    rw [← zpow_add₀ (by norm_num : (10 : ℚ) ≠ 0)]; simp
  have halign : (decimalShift e : ℤ) + 1 - pe = e :=
    align_all e (by simpa [Finset.mem_Icc] using he)
  have hsh : decimalShift e < 4 := decimal_shift_lt_four e he
  calc (10 : ℚ) ^ (-k) * 2 ^ (128 - pe) * (2 ^ (1 - e) * 10 ^ k)
      = (10 ^ (-k) * 10 ^ k) * (2 ^ (128 - pe) * 2 ^ (1 - e)) := by
        ring
    _ = (2 : ℚ) ^ ((128 - pe) + (1 - e)) := by
        rw [h10, one_mul, ← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
    _ = 2 ^ (128 - decimalShift e) := by
        rw [show (128 - pe) + (1 - e) = ((128 - decimalShift e : ℕ) : ℤ)
              from by omega, zpow_natCast]

-- `trimMul` clears the denominator in `power10_exact_ratio`, leaving
-- `trimNum` times the binary-decimal scaling factor.
theorem trim_mul_eq (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    (trimMul e : ℚ)
      = (trimNum e : ℚ) * (2 ^ (1 - e) * 10 ^ decimalExponent e) := by
  have hd : (0 : ℚ) < (trimDen e : ℚ) := by
    exact_mod_cast trim_den_pos e
  have hnum :
      (10 : ℚ) ^ (-decimalExponent e)
          * 2 ^ (128 - power10Exponent (-decimalExponent e))
          * trimDen e
        = trimNum e := by
    rw [power10_exact_ratio, ← trimNum, ← trimDen,
      div_mul_cancel₀ _ (ne_of_gt hd)]
  rw [trimMul]
  push_cast
  rw [← exact_scale e he, ← hnum]
  ring

-- Scaling half a ULP in decimal-significand space by `trimMul` gives `trimNum`.
theorem trim_mul_ulp (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    let k := decimalExponent e
    ulp e * (10 ^ k)⁻¹ / 2 * (trimMul e : ℚ) = (trimNum e : ℚ) := by
  intro k
  calc ulp e * (10 ^ k)⁻¹ / 2 * (trimMul e : ℚ)
      = (trimNum e : ℚ) * (2 ^ e * 2 ^ (1 - e) / 2)
          * ((10 ^ k)⁻¹ * 10 ^ k) := by
        rw [ulp, trim_mul_eq e he]; ring
    _ = (trimNum e : ℚ) := by
        rw [← zpow_add₀ (two_ne_zero' ℚ) e (1 - e),
          show e + (1 - e) = 1 from by ring, zpow_one,
          inv_mul_cancel₀ (by positivity)]
        ring

theorem trim_mul_value (f : ℕ) (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    let k := decimalExponent e
    value f e * (10 ^ k)⁻¹ * (trimMul e : ℚ) = 2 * f * (trimNum e : ℚ) := by
  rw [← trim_mul_ulp e he, value, ulp]
  ring

-- The scale sends the decimal step `10` to the window modulus.
theorem trim_mul_ten (e : ℤ) :
    (10 : ℚ) * (trimMul e : ℚ) = (trimScale e : ℚ) := by
  rw [trimScale, trimModulus, trimMul]
  push_cast
  ring

-- The lower multiple-of-ten candidate is exactly `trimGap` below the scaled
-- value: `trim_residue_add_ten` gives the main term, and `trim_num_split`
-- accounts for the remaining low bits.
theorem trim_mul_dec_ten (f : ℕ) (e : ℤ) (hsh : decimalShift e < 4) :
    (sigHi f e - sigHi f e % 10) * trimMul e + trimGap f e
      = 2 * f * trimNum e := by
  calc (sigHi f e - sigHi f e % 10) * trimMul e + trimGap f e
      = ((sigHi f e - sigHi f e % 10) * 2 ^ (128 - decimalShift e)
            + trimResidue f e) * trimDen e
          + 2 * f * (trimNum e % trimDen e) := by
        rw [trimMul, trimGap]; ring
    _ = 2 * f * (trimDen e * trimSig e + trimNum e % trimDen e) := by
        rw [trim_residue_add_ten f e hsh]; ring
    _ = 2 * f * trimNum e := by rw [trim_num_split]

-- Rational form of `trim_mul_dec_ten`, rearranged as a scaled distance.
theorem trim_mul_dec_ten_rat (f : ℕ) (e : ℤ)
    (hsh : decimalShift e < 4) :
    ((sigHi f e - sigHi f e % 10 : ℕ) : ℚ) * trimMul e
      = 2 * f * trimNum e - trimGap f e := by
  have hn := congrArg (fun n : ℕ => (n : ℚ)) (trim_mul_dec_ten f e hsh)
  push_cast at hn
  linarith

-- A candidate at scaled distance `dist` from the value is within half a ULP
-- whenever `|dist| ≤ trimNum`, strictly so when `|dist| < trimNum`.
theorem trim_half_ulp {cand dist : ℚ} (f : ℕ) (e : ℤ)
    (he : -1074 ≤ e ∧ e ≤ 971)
    (hscale : cand * (trimMul e : ℚ) = 2 * f * (trimNum e : ℚ) - dist) :
    let k := decimalExponent e
    let x := value f e * (10 ^ k)⁻¹
    let u := ulp e * (10 ^ k)⁻¹
    (|dist| ≤ (trimNum e : ℚ) → |cand - x| ≤ u / 2) ∧
      (|dist| < (trimNum e : ℚ) → |cand - x| < u / 2) := by
  intro k x u
  have hpos : (0 : ℚ) < (trimMul e : ℚ) :=
    Nat.cast_pos.mpr
      (by rw [trimMul]; exact Nat.mul_pos (by positivity) (trim_den_pos e))
  have habs := abs_mul (cand - x) (trimMul e : ℚ)
  rw [abs_of_pos hpos] at habs
  have hdist : |cand - x| * (trimMul e : ℚ) = |dist| := by
    rw [← habs, sub_mul, trim_mul_value f e he, hscale,
      show 2 * (f : ℚ) * (trimNum e : ℚ) - dist
        - 2 * f * (trimNum e : ℚ) = -dist from by ring,
      abs_neg]
  exact ⟨fun hle =>
      le_of_mul_le_mul_right (by rw [hdist, trim_mul_ulp e he]; exact hle) hpos,
    fun hlt =>
      lt_of_mul_lt_mul_right
        (by rw [hdist, trim_mul_ulp e he]; exact hlt) hpos.le⟩

/-! ### The multiple-of-ten candidates -/

-- The trim-down candidate is in range whenever `roundD0` fires.
theorem dec_ten_down (f : ℕ) (e : ℤ) (h : Regular f e)
    (hd0 : (toDecimalCandidates f e).roundD0 = true) :
    let k := decimalExponent e
    let ten : ℕ := sigHi f e - sigHi f e % 10
    let x := value f e * (10 ^ k)⁻¹
    let u := ulp e * (10 ^ k)⁻¹
    if f % 2 = 0 then
      |(ten : ℚ) - x| ≤ u / 2
    else
      |(ten : ℚ) - x| < u / 2 := by
  intro k ten x u
  have hsh : decimalShift e < 4 := decimal_shift_lt_four e h.2.2
  obtain ⟨hle, hlt⟩ := trim_half_ulp f e h.2.2 (trim_mul_dec_ten_rat f e hsh)
  have habs : |(trimGap f e : ℚ)| = (trimGap f e : ℚ) :=
    abs_of_nonneg (Nat.cast_nonneg _)
  -- yy's packed operands corresponding to `c` and `halfUlp`.
  set w := trimResidue f e / trimUnit e with hw
  set p := trimSig e / trimUnit e with hp
  -- The two yy tests, in the window quantities they actually compare.
  have hd0' :
      (if p = w then decide (f % 2 = 0) else decide (w < p)) = true := by
    rw [hw, hp, ← trim_c_eq f e hsh, ← trim_half_ulp_eq e hsh]
    exact hd0
  split at hd0'
  -- The apparent tie `halfUlp = c`, which yy takes only for even `f`.
  · rename_i htie
    simp only [show f % 2 = 0 from by simpa using hd0', reduceIte]
    exact hle (by
      rw [habs]
      exact_mod_cast trim_gap_le f e h htie.symm.le)
  -- The strict comparison `c < halfUlp`.
  · have hd := hlt (by
      rw [habs]
      exact_mod_cast trim_gap_lt f e h (by simpa using hd0'))
    split_ifs
    · exact hd.le
    · exact hd

-- The trim-up candidate is in range whenever `roundU0` fires. All three yy
-- branches leave the packed sum within one window unit of the modulus; only the
-- final `t0 ≤ t1` branch can fire for odd `f`, and there the bound is strict.
theorem dec_ten_up (f : ℕ) (e : ℤ) (h : Regular f e)
    (hu0 : (toDecimalCandidates f e).roundU0 = true) :
    let k := decimalExponent e
    let ten : ℕ := sigHi f e - sigHi f e % 10
    let x := value f e * (10 ^ k)⁻¹
    let u := ulp e * (10 ^ k)⁻¹
    if f % 2 = 0 then
      |(ten : ℚ) + 10 - x| ≤ u / 2
    else
      |(ten : ℚ) + 10 - x| < u / 2 := by
  intro k ten x u
  have hsh : decimalShift e < 4 := decimal_shift_lt_four e h.2.2
  obtain ⟨hle, hlt⟩ := trim_half_ulp f e h.2.2 (cand := (ten : ℚ) + 10)
    (dist := (trimGap f e : ℚ) - (trimScale e : ℚ)) (by
      rw [add_mul, trim_mul_dec_ten_rat f e hsh, trim_mul_ten e]; ring)
  -- The free side, shared by all branches.
  have hfree : (trimGap f e : ℚ) - (trimScale e : ℚ) < (trimNum e : ℚ) := by
    have hz : (trimGap f e : ℚ) < (trimScale e : ℚ) + (trimNum e : ℚ) := by
      exact_mod_cast trim_gap_lt_scale_add f e h
    linarith
  -- Convert the integer bounds to scaled error bounds.
  have hle_of_pack (hpack : trimScale e ≤ trimGap f e + trimNum e) :
      |(ten : ℚ) + 10 - x| ≤ u / 2 := by
    refine hle (abs_le.mpr ⟨?_, hfree.le⟩)
    have hz : (trimScale e : ℚ) ≤ (trimGap f e : ℚ) + (trimNum e : ℚ) := by
      exact_mod_cast hpack
    linarith
  have hlt_of_pack (hpack : trimScale e < trimGap f e + trimNum e) :
      |(ten : ℚ) + 10 - x| < u / 2 := by
    refine hlt (abs_lt.mpr ⟨?_, hfree⟩)
    have hz : (trimScale e : ℚ) < (trimGap f e : ℚ) + (trimNum e : ℚ) := by
      exact_mod_cast hpack
    linarith
  -- yy's two packed operands, `t1 = w + p` against the modulus `t0 = 10·2^60`.
  set w := trimResidue f e / trimUnit e with hw
  set p := trimSig e / trimUnit e with hp10
  -- Lift `t0 ≤ t1 + 1` to the untruncated bound used by `trim_scale_le`.
  have hpack_of_ge (hb : 10 * 2 ^ 60 ≤ w + p + 1) :
      trimModulus e ≤ trimResidue f e + trimSig e + trimUnit e := by
    rw [trim_modulus_eq e hsh]
    calc trimUnit e * (10 * 2 ^ 60)
        ≤ trimUnit e * (w + p) + trimUnit e := by
          rw [← Nat.mul_succ]; exact Nat.mul_le_mul_left _ hb
      _ ≤ trimResidue f e + trimSig e + trimUnit e :=
          Nat.add_le_add_right (trim_pack f e (w + p) le_rfl) _
  -- The three yy tests, in the window quantities they actually compare.
  have hu0' : (if w + p + 1 = 10 * 2 ^ 60 then decide (f % 2 = 0)
      else if k = 0 ∧ w + p = 10 * 2 ^ 60 then decide (f % 2 = 0)
      else decide (10 * 2 ^ 60 ≤ w + p)) = true := by
    rw [hw, hp10, ← trim_c_eq f e hsh, ← trim_half_ulp_eq e hsh]
    exact hu0
  split at hu0'
  -- The one-LSB-offset tie `t1 + 1 = t0`, for even `f` only.
  · rename_i htie1
    simp only [show f % 2 = 0 from by simpa using hu0', reduceIte]
    exact hle_of_pack (trim_scale_le f e h (hpack_of_ge htie1.ge))
  · split at hu0'
    -- The exact tie `t1 = t0`, accepted only when `k = 0`, even `f` only.
    · rename_i htie0
      simp only [show f % 2 = 0 from by simpa using hu0', reduceIte]
      exact hle_of_pack
        (trim_scale_le f e h (hpack_of_ge (Nat.le_succ_of_le htie0.2.ge)))
    -- The final `t0 ≤ t1` branch, the only one that can fire for odd `f`.
    · rename_i hnot_tie0
      have hplain : 10 * 2 ^ 60 ≤ w + p := by simpa using hu0'
      by_cases heven : f % 2 = 0
      · simp only [heven, reduceIte]
        exact hle_of_pack
          (trim_scale_le f e h (hpack_of_ge (Nat.le_succ_of_le hplain)))
      · simp only [heven, reduceIte]
        exact hlt_of_pack (trim_scale_lt f e h hplain hnot_tie0)

/-! ### The main theorems -/

-- The decimal significand produced by yy is within half a scaled ULP
-- of the exact value, with equality allowed only when f is even.
theorem decimal_significand_error_bound
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
  simp only [zpow_neg]
  let x := value f e * (10 ^ c.k)⁻¹
  let u := ulp e * (10 ^ c.k)⁻¹
  let ten := sigHi f e - sigHi f e % 10
  let InRange (dec : ℕ) : Prop :=
    if f % 2 = 0 then
      |dec - x| ≤ u / 2
    else
      |dec - x| < u / 2
  have hone : InRange c.decOne := by
    have hs := dec_one_error_bound f e h
    simp only [InRange, c, x, u]
    split_ifs <;> linarith [hs]
  have hdecTen :
      c.decTen = ten + (if c.roundU0 then 10 else 0) := rfl
  have hk : c.k = decimalExponent e := rfl
  -- `roundU0` determines which multiple-of-ten candidate `decTen` denotes;
  -- whether both trimming flags can fire is irrelevant.
  have hten_d0 (hu0 : c.roundU0 = false) (hd0 : c.roundD0 = true) :
      InRange c.decTen := by
    rw [hdecTen, hu0]
    simpa [InRange, ten, x, u, hk] using dec_ten_down f e h hd0
  have hten_u0 (hu0 : c.roundU0 = true) : InRange c.decTen := by
    rw [hdecTen, hu0]
    simpa [InRange, ten, x, u, hk] using dec_ten_up f e h hu0
  cases hu0 : c.roundU0
  · cases hd0 : c.roundD0 <;> simp_all [InRange, x, u]
  · simp_all [InRange, x, u]

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
  -- Rescale the significand error bound back to the original value.
  have hdist := decimal_significand_error_bound f e h
  simp only [Roundtrips]
  split_ifs with heven <;>
    rw [← hrescale_error, abs_mul, abs_of_pos hp, ← hscale]
  · exact mul_le_mul_of_nonneg_right
      (by simpa [hdk, heven] using hdist) (le_of_lt hp)
  · exact mul_lt_mul_of_pos_right
      (by simpa [hdk, heven] using hdist) hp
