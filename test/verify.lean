import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.Tactic

-- The finite checks below enumerate up to 2046 exponents in the kernel.
-- Range-wide `decide` checks raise the recursion guard where they appear,
-- and those evaluating 10^324 also raise the elaborator's exponentiation
-- guard.

/-! ### The specification -/

-- Exact rational value represented by binary significand `f` and exponent `e`.
def value (f : ℕ) (e : ℤ) : ℚ := f * 2 ^ e

-- One ULP for a regularly spaced value with exponent `e`.
def ulp (e : ℤ) : ℚ := 2 ^ e

-- Whether the exact rational result `r` rounds to the regularly spaced value
-- f · 2^e under round-to-nearest, ties-to-even.
def Roundtrips (f : ℕ) (e : ℤ) (r : ℚ) : Prop :=
  if f % 2 = 0 then
    |r - value f e| ≤ ulp e / 2
  else
    |r - value f e| < ulp e / 2

-- A decimal representation is shortest if it round-trips and no value on the
-- next coarser decimal grid does. The grids are nested, so refuting the next
-- one refutes every coarser one. It also forces `d` to have no trailing zero,
-- since `d / 10` at `k + 1` would denote the same value.
def Shortest (f : ℕ) (e : ℤ) (d : ℕ) (k : ℤ) : Prop :=
  Roundtrips f e (d * 10 ^ k) ∧
    ∀ d' : ℕ, ¬Roundtrips f e (d' * 10 ^ (k + 1))

-- Whether f · 2^e is a regularly spaced normal binary64 value,
-- excluding powers of 2.
def Regular (f : ℕ) (e : ℤ) : Prop :=
  2 ^ 52 < f ∧ f < 2 ^ 53 ∧
   -1074 ≤ e ∧ e ≤ 971

/-! ### Decimal reduction -/

-- Removes trailing zeros from a decimal significand, shifting the exponent to
-- preserve the represented value.
def reduceDecimal (dec : ℕ × ℤ) : ℕ × ℤ :=
  if 0 < dec.1 ∧ dec.1 % 10 = 0 then reduceDecimal (dec.1 / 10, dec.2 + 1)
  else dec
termination_by dec.1
decreasing_by omega

-- Reduction leaves either zero or a significand with no trailing zero.
theorem reduce_reduced (dec : ℕ × ℤ) :
    (reduceDecimal dec).1 = 0 ∨ (reduceDecimal dec).1 % 10 ≠ 0 := by
  fun_induction reduceDecimal dec with
  | case1 dec _ ih => exact ih
  | case2 dec hstop => omega

-- Reduction shifts the exponent by the number of zeros stripped and removes
-- the corresponding power of ten from the significand.
theorem reduce_shift (dec : ℕ × ℤ) :
    ∃ t : ℕ, (reduceDecimal dec).2 = dec.2 + t
      ∧ dec.1 = (reduceDecimal dec).1 * 10 ^ t := by
  fun_induction reduceDecimal dec with
  | case1 dec hgo ih =>
    obtain ⟨t, hk, hd⟩ := ih
    dsimp only at hk hd
    refine ⟨t + 1, by push_cast; omega, ?_⟩
    rw [pow_succ, ← Nat.mul_assoc, ← hd]
    omega
  | case2 dec _ => exact ⟨0, by simp, by simp⟩

-- Trailing zeros can move between the significand and the exponent.
theorem ten_pow_shift (d t : ℕ) (k : ℤ) :
    ((d * 10 ^ t : ℕ) : ℚ) * 10 ^ k = (d : ℚ) * 10 ^ (k + (t : ℤ)) := by
  push_cast
  rw [zpow_add₀ (by norm_num : (10 : ℚ) ≠ 0), zpow_natCast]
  ring

-- Reduction is value-preserving.
theorem reduce_value (dec : ℕ × ℤ) :
    let (d, k) := reduceDecimal dec
    (d : ℚ) * 10 ^ k = (dec.1 : ℚ) * 10 ^ dec.2 := by
  obtain ⟨t, hkt, hstrip⟩ := reduce_shift dec
  rcases hred : reduceDecimal dec with ⟨d, k⟩
  simp only [hred] at hkt hstrip
  rw [hstrip, hkt, ten_pow_shift]

/-! ### The truncated power of ten -/

-- Binary exponent of 10^k used to normalize its 128-bit significand.
def power10Exponent (k : ℤ) : ℤ :=
  if 0 ≤ k then
    Nat.log 2 (10 ^ k.toNat) + 1
  else
    -Nat.log 2 (10 ^ (-k).toNat)

-- Truncated 128-bit normalized binary significand of 10^k.
def power10Significand (k : ℤ) : ℕ :=
  ⌊(10 : ℚ) ^ k * 2 ^ (128 - power10Exponent k)⌋₊

-- Numerator and denominator of the exact scaled power of ten `10^k·2^(128-pe)`,
-- with negative exponents moved to the denominator. Writing it as a ratio of
-- naturals turns the truncation into a single `Nat` division, so the
-- exponent-wise checks below can run in the kernel.
def power10Num (k : ℤ) : ℕ :=
  10 ^ k.toNat * 2 ^ (128 - power10Exponent k).toNat

def power10Den (k : ℤ) : ℕ :=
  10 ^ (-k).toNat * 2 ^ (power10Exponent k - 128).toNat

-- `power10Exponent` is characterized by this interval: it is the binary
-- exponent that normalizes `10^k`. Both bounds are the defining bounds of
-- `Nat.log`; for `k < 0` inversion exchanges them, so there the log's lower
-- bound has to be strict, which it is because a power of two carries no five.
theorem power10_exponent_bounds (k : ℤ) :
    (2 : ℚ) ^ (power10Exponent k - 1) ≤ 10 ^ k ∧
      (10 : ℚ) ^ k < 2 ^ power10Exponent k := by
  unfold power10Exponent
  split_ifs with hk
  · set n := k.toNat with hn
    set l := Nat.log 2 (10 ^ n)
    have h10 : (10 : ℚ) ^ k = (10 : ℚ) ^ n := by
      rw [← zpow_natCast (10 : ℚ) n, hn, Int.toNat_of_nonneg hk]
    refine ⟨?_, ?_⟩
    · rw [h10, show (l : ℤ) + 1 - 1 = ((l : ℕ) : ℤ) from by ring, zpow_natCast]
      exact_mod_cast Nat.pow_log_le_self 2 (by positivity : 10 ^ n ≠ 0)
    · rw [h10, show (l : ℤ) + 1 = ((l + 1 : ℕ) : ℤ) from by omega,
        zpow_natCast]
      exact_mod_cast Nat.lt_pow_succ_log_self (by norm_num) (10 ^ n)
  · set m := (-k).toNat with hm
    set l := Nat.log 2 (10 ^ m)
    have hk' : k = -(m : ℤ) := by omega
    refine ⟨?_, ?_⟩
    · rw [hk', zpow_neg, zpow_natCast,
        show (-(l : ℤ) - 1) = -((l + 1 : ℕ) : ℤ) from by omega,
        zpow_neg, zpow_natCast, inv_le_inv₀ (by positivity) (by positivity)]
      exact_mod_cast (Nat.lt_pow_succ_log_self (by norm_num) (10 ^ m)).le
    · have hne : (2 : ℕ) ^ l ≠ 10 ^ m := by
        intro hcon
        have h5 : (5 : ℕ) ∣ 2 ^ l := by
          rw [hcon]
          exact dvd_pow (⟨2, rfl⟩ : (5 : ℕ) ∣ 10) (by omega)
        have := Nat.prime_five.dvd_of_dvd_pow h5
        omega
      rw [hk', zpow_neg, zpow_natCast, zpow_neg, zpow_natCast,
        inv_lt_inv₀ (by positivity) (by positivity)]
      exact_mod_cast lt_of_le_of_ne (Nat.pow_log_le_self 2 (by positivity)) hne

-- Scaling that interval by `2^(128-pe)` makes the significand a normalized
-- 128-bit number: its top bit is set, which is what makes `power10Exponent`
-- an exponent for a 128-bit significand, and it still fits in 128 bits.
theorem power10_significand_bounds (k : ℤ) :
    2 ^ 127 ≤ power10Significand k ∧ power10Significand k < 2 ^ 128 := by
  refine ⟨?_, ?_⟩
  · have hx : (2 : ℚ) ^ (127 : ℕ) ≤ 10 ^ k * 2 ^ (128 - power10Exponent k) := by
      have hmul := mul_le_mul_of_nonneg_right (power10_exponent_bounds k).1
        (by positivity : (0 : ℚ) ≤ 2 ^ (128 - power10Exponent k))
      rw [← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0),
        show power10Exponent k - 1 + (128 - power10Exponent k) = (127 : ℤ) from
          by ring] at hmul
      simpa using hmul
    exact Nat.le_floor (by push_cast; exact hx)
  · have hx : (10 : ℚ) ^ k * 2 ^ (128 - power10Exponent k) < 2 ^ (128 : ℕ) := by
      have hmul := mul_lt_mul_of_pos_right (power10_exponent_bounds k).2
        (by positivity : (0 : ℚ) < 2 ^ (128 - power10Exponent k))
      rw [← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0),
        show power10Exponent k + (128 - power10Exponent k) = (128 : ℤ) from by
          ring] at hmul
      simpa using hmul
    exact (Nat.floor_lt (by positivity)).mpr (by push_cast; exact hx)

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

/-! ### yy's conversion -/

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

/-! ### Exponent alignment -/

-- The shift used by yy's regular path is less than 4.
theorem decimal_shift_lt_four (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    decimalShift e < 4 := by
  unfold decimalShift decimalExponent
  omega

section

-- The checks compute powers as large as `10^324`.
set_option exponentiation.threshold 5000
set_option maxRecDepth 100000

-- Exponent alignment over the binary64 exponent range.
theorem align_all :
    ∀ e ∈ Finset.Icc (-1074 : ℤ) 971,
      (decimalShift e : ℤ) + 1 - power10Exponent (-decimalExponent e) = e := by
  decide

-- Away from `k = 0`, the normalized exact power of ten clears `2^127` by a wide
-- margin: `10^k` is never within `2^-62` of a power of two, only within about
-- `2^-10` of one. How close it comes is not a magnitude property, so it is
-- checked per decimal exponent.
theorem power10_margin_all :
    ∀ k ∈ Finset.Icc (-324 : ℤ) 292,
      k = 0 ∨ (2 ^ 127 + 2 ^ 65) * power10Den (-k) ≤ power10Num (-k) := by
  decide

end

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

-- yy produces two candidates from the same product `2·f·p10`, differing only
-- in the step `m` between admissible values: `2^(128-h)` for `sigHi` and ten
-- times that for its multiple of ten. In both, the candidate is the quotient
-- `2·f·p10 / m` and `stepResidue` is the exact remainder above it, in units of
-- `2^(h-128)` of the scaled value.
def stepResidue (m f : ℕ) (e : ℤ) : ℕ := 2 * f * trimSig e % m

-- The remainder above the multiple-of-ten candidate:
-- `ten·2^(128-h) + trimResidue = 2·f·p10`.
def trimResidue (f : ℕ) (e : ℤ) : ℕ := stepResidue (trimModulus e) f e

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

-- A power of two splits into the shift and what is left of the window.
theorem pow_shift_split (e : ℤ) (n : ℕ) (hn : decimalShift e ≤ n) :
    (2 : ℕ) ^ n = 2 ^ decimalShift e * 2 ^ (n - decimalShift e) := by
  rw [← pow_add]
  congr 1
  omega

-- The same split with the extra factor of ten of the window modulus.
theorem pow_split (e : ℤ) (hsh : decimalShift e < 4) :
    (2 : ℕ) ^ 128 * 10 = 2 ^ decimalShift e * trimModulus e := by
  rw [trimModulus, pow_shift_split e 128 (by omega)]
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

-- `sigHi` is the quotient at the unit step. The multiple-of-ten candidate is
-- ten times the quotient at the window step.
theorem sig_hi_quotient (f : ℕ) (e : ℤ) (hsh : decimalShift e < 4) :
    sigHi f e = 2 * f * trimSig e / 2 ^ (128 - decimalShift e) := by
  rw [sig_hi_eq, pow_shift_split e 128 (by omega),
    Nat.mul_div_mul_left _ _ (by positivity)]

theorem sig_hi_ten_quotient (f : ℕ) (e : ℤ) (hsh : decimalShift e < 4) :
    sigHi f e - sigHi f e % 10
      = 10 * (2 * f * trimSig e / trimModulus e) := by
  -- Dividing the unit-step quotient by ten is dividing by the window step.
  have hdiv : sigHi f e / 10 = 2 * f * trimSig e / trimModulus e := by
    rw [sig_hi_quotient f e hsh, Nat.div_div_eq_div_mul,
      show 2 ^ (128 - decimalShift e) * 10 = trimModulus e from by
        rw [trimModulus]; ring]
  have hmod := Nat.div_add_mod (sigHi f e) 10
  rw [← hdiv]
  omega

/-! ### What the rounding certificates say about the window

`roundD0` compares `W / U` with `p10 / U`, while `roundU0` compares their sum
with `N / U`. These tests see the exact quantities only up to one window unit
`U`; `dec_ten_down` and `dec_ten_up` translate their outcomes back into exact
bounds.

Because truncation is one-sided, the soundness directions are asymmetric: the
plain `roundU0` test is safe whenever it fires, while the one-LSB-offset test
`t1 + 1 = t0` and the trim-down tests may accept one unit early.

Completeness reads the same comparisons in the opposite direction. A test that
does not fire bounds the exact gap from the other side, again with at most one
unit of uncertainty. There the relevant margin is the distance from the
discarded low bits of `p10` to the next window boundary, rather than the low
bits themselves; this is why `trimLowBitsHolds` certifies both sides of the
unit.
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
it has not wrapped, which the packed comparisons guarantee. Each bound is needed
in both directions, so each of the two boundaries contributes a window on either
side of it; the boundary value itself lies in none of them, which is what leaves
the exact ties above their room. A violation would put a single modular
progression inside an interval of width below `den·U`, a window of relative
width `U/N ≈ 2^(-63.3)`. The finite certificates rule this out for every
exponent. This is where verify.py counts solutions with `floor_sum`; here a
refutation certificate reaches the same conclusion with one small check per
window.
-/

-- The exact distance from a candidate to the scaled value, with the
-- denominator of the exact power of ten cleared: the residue contributes
-- `den·W` and the truncation `p10Exact - p10 = τ/den` costs `2·f·τ`.
def stepGap (m f : ℕ) (e : ℤ) : ℕ :=
  trimDen e * stepResidue m f e + 2 * f * (trimNum e % trimDen e)

-- The distance to the multiple-of-ten candidate, and the window modulus with
-- the same denominator cleared.
def trimGap (f : ℕ) (e : ℤ) : ℕ := stepGap (trimModulus e) f e

def trimScale (e : ℤ) : ℕ := trimModulus e * trimDen e

-- How far a gap can be from the multiple-of-ten candidate and still be accepted
-- by a packed comparison: `den·(p10 + U)`. Written as `num / den` rather than
-- `trimSig` so the certificate remains purely natural-number computation.
def trimBnd (e : ℤ) : ℕ := trimDen e * (trimNum e / trimDen e + trimUnit e)

-- One window unit with the denominator cleared: the resolution of the packed
-- comparison and the scale of the narrow windows refuted below.
def trimEdge (e : ℤ) : ℕ := trimUnit e * trimDen e

-- Integer form of `2·p10 + 2 ≤ N`, where `N = trimModulus`. The truncation is
-- expressed as natural division so the finite check reduces directly in the
-- kernel.
def trimNarrowHolds (e : ℤ) : Bool :=
  decide (2 * (trimNum e / trimDen e) + 2 ≤ trimModulus e)

set_option exponentiation.threshold 5000 in
set_option maxRecDepth 100000 in
theorem trim_narrow_all :
    ∀ e ∈ Finset.Icc (-1074 : ℤ) 971, trimNarrowHolds e = true := by decide

-- Integer form of `2^54·(p10Exact - p10) ≤ p10 % U` and of its complement
-- `2^54·(p10Exact - p10) ≤ U - p10 % U`, with the denominator cleared. The
-- truncation error fits in the bits the packed comparison discards, measured
-- from either end of the window unit.
def trimLowBitsHolds (e : ℤ) : Bool :=
  let num := trimNum e
  let den := trimDen e
  let low := num / den % trimUnit e
  decide (2 ^ 54 * (num % den) ≤ low * den
    ∧ 2 ^ 54 * (num % den) + low * den ≤ trimUnit e * den)

set_option exponentiation.threshold 5000 in
set_option maxRecDepth 100000 in
theorem trim_low_bits_all :
    ∀ e ∈ Finset.Icc (-1074 : ℤ) 971, trimLowBitsHolds e = true := by decide

-- The discarded low bits `p10 % U` dominate the cached-power truncation error
-- `p10Exact - p10 = τ/den`: the margin used when a packed comparison is strict
-- on the low side. Which bits of the power of ten survive truncation is not a
-- magnitude property, so this is checked per exponent.
theorem trim_low_bits (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    2 ^ 54 * (trimNum e % trimDen e)
      ≤ trimSig e % trimUnit e * trimDen e := by
  have hcert := trim_low_bits_all e (by simpa [Finset.mem_Icc] using he)
  simp only [trimLowBitsHolds, decide_eq_true_eq] at hcert
  rw [trim_sig_nat]
  exact hcert.1

-- The complementary distance to the next window unit dominates the same
-- truncation error: the margin used when a packed comparison falls short of
-- its boundary, which is what the completeness directions need.
theorem trim_high_bits (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    2 ^ 54 * (trimNum e % trimDen e) + trimSig e % trimUnit e * trimDen e
      ≤ trimUnit e * trimDen e := by
  have hcert := trim_low_bits_all e (by simpa [Finset.mem_Icc] using he)
  simp only [trimLowBitsHolds, decide_eq_true_eq] at hcert
  rw [trim_sig_nat]
  exact hcert.2

/-! ### Refuting the trim windows

A Nadezhin-style separation proof, as used in Schubfach, was considered but
still requires a finite Diophantine check over the binary64 significand range.
The direct modular-window formulation below matches yy more closely and is
substantially simpler.

Soundness and completeness for both trim boundaries reduce to one arithmetic
question per exponent. Writing `num/den` for the exact power of ten,
`modulus = N·den`, and `g = 2·num`, any violation forces `g·f mod modulus` into
a narrow interval of width below `den·U`.

A window is refuted by a single multiplier `q`. Writing `y = g·f - modulus·j`
for the residue and `r = g·q - modulus·p` for the approximation error of
`p/q ≈ g/modulus`, one has

    modulus·(p·f - q·j) = q·y - f·r.

Thus, if `q·y - f·r` stays strictly between two consecutive multiples of
`modulus` throughout the box `f ∈ [F₀,F₁]`, `y ∈ [lo,hi]`, no `f` can hit the
window. Continued-fraction denominators of `g/modulus` make `r` small enough
that such a separating multiplier is easy to find for these windows.

The multiplier is a witness, not an assumption: the search procedure is
untrusted, and the kernel proves correctness by checking the certificate it
returns.
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

-- A list of windows refuted by one multiplier, over the significand range.
def modWindowsRefuted (g modulus q : ℤ) (windows : List (ℤ × ℤ)) : Bool :=
  windows.all fun w =>
    modWindowRefuted g modulus (2 ^ 52 + 1) (2 ^ 53 - 1) w.1 w.2 q

-- The four windows a violation would have to hit around `num` and
-- `modulus - num`. The outward windows extend to `bnd`, while the inward
-- windows extend by one `edge`; soundness uses the outward pair and
-- completeness the inward pair. The trim quantities are computed once here so
-- the search below can retry a window without recomputing the power of ten.
def trimWindows (e : ℤ) : List (ℤ × ℤ) :=
  let num : ℤ := trimNum e
  let bnd : ℤ := trimBnd e
  let edge : ℤ := trimEdge e
  let modulus : ℤ := trimScale e
  [(num + 1, bnd - 1), (num - edge + 1, num - 1),
    (modulus - bnd, modulus - num - 1),
    (modulus - num + 1, modulus - num + edge - 1)]

-- The multiplier is searched for rather than tabulated, by the elaborator
-- rather than by the kernel. Convergent denominators of `g/modulus` give the
-- relevant best rational approximations. The Euclidean remainder `v` that
-- produces `qc` is the error
-- `|g·qc - modulus·p|`; it must fall below `modulus/2^53` before a window can
-- be refuted. Testing that first skips the small denominators and leaves at
-- most three window checks per exponent. Such a denominator leaves the span of
-- `q·y - f·r` at about `2·√(n·(hi-lo)/modulus) ≈ 2⁻⁵` of `modulus`, where
-- `n = 2^52` is the number of significands in the box, so it fits between
-- consecutive multiples with room to spare.
private def modCertSearch (g modulus : ℤ) (windows : List (ℤ × ℤ)) :
    ℕ → ℤ → ℤ → ℤ → ℤ → ℤ
  | 0, _, _, _, qc => qc
  | n + 1, u, v, qp, qc =>
    if decide (2 ^ 53 * v < modulus)
        && modWindowsRefuted g modulus qc windows then
      qc
    else if v = 0 then
      qc
    else
      modCertSearch g modulus windows n v (u % v) qc (u / v * qc + qp)

-- All windows of one exponent, refuted by a given multiplier: a handful of
-- big-integer operations, with the search for the multiplier left outside.
def trimCertificateValid (e q : ℤ) : Bool :=
  modWindowsRefuted (2 * trimNum e) (trimScale e) q (trimWindows e)

/-! #### Certifying the exponent range

Finding a multiplier and checking one are separated along the trust boundary.
`findTrimCertificate` runs during elaboration, outside the proof term. The
tactic below quotes the multiplier it returns as an integer literal, and the
kernel then checks `trimCertificateValid` on that literal. The search procedure
is untrusted: if it produces a bad multiplier, the kernel check fails. Thus only
the certificate appears in the proof term, not the computation that found it.

Previously the search itself was reduced by the kernel through `decide`, which
kept the intermediate search computation alive while checking the declaration
and made this section expensive.
-/

private def findTrimCertificate (e : ℤ) : ℤ :=
  let g : ℤ := 2 * trimNum e
  let modulus : ℤ := trimScale e
  modCertSearch g modulus (trimWindows e) 80 modulus (g % modulus) 0 1

open Lean Elab Tactic Meta in
/-- Close a goal `∃ q, trimCertificateValid e q = true` for a literal exponent
`e` by searching for a multiplier and quoting it into the proof term. -/
elab "trim_cert" : tactic => do
  let target ← whnfR (← (← getMainGoal).getType)
  let_expr Exists _ pred := target
    | throwError "trim_cert: expected `∃ q, trimCertificateValid e q = true`"
  let_expr Eq _ lhs _ := pred.bindingBody!
    | throwError "trim_cert: expected an equation under the existential"
  let_expr trimCertificateValid e _ := lhs
    | throwError "trim_cert: expected `trimCertificateValid e q`, got {lhs}"
  let some exponent := e.int?
    | throwError "trim_cert: the exponent {e} is not a literal"
  let q ← Term.exprToSyntax (toExpr (findTrimCertificate exponent))
  evalTactic (← `(tactic| exact ⟨$q, by decide +kernel⟩))

theorem trim_windows_refuted (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    ∃ q, trimCertificateValid e q = true := by
  obtain ⟨hlo, hhi⟩ := he
  interval_cases e <;> trim_cert

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
  rw [trimGap, stepGap, stepResidue, trim_sig_nat, trimScale] at hlt ⊢
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
  have hp_lt : trimSig e < 2 ^ 128 := (power10_significand_bounds _).2
  have hgap : (2 : ℕ) ^ 128 + 2 ^ 68 ≤ 10 * 2 ^ 125 := by norm_num
  have hwindow : trimSig e + trimUnit e ≤ trimModulus e := by omega
  rw [trimScale, Nat.mul_comm (trimModulus e)]
  exact Nat.mul_le_mul_left _ hwindow

-- `p10Exact ≥ 2^127`, with the denominator cleared.
theorem trim_num_lower (e : ℤ) : 2 ^ 127 * trimDen e ≤ trimNum e := by
  have h : 2 ^ 127 ≤ trimNum e / trimDen e := by
    rw [← trim_sig_nat]; exact (power10_significand_bounds _).1
  exact (Nat.le_div_iff_mul_le (trim_den_pos e)).mp h

-- The resolution of the packed comparison is negligible against the power of
-- ten: `U ≤ 2^68` while `p10Exact ≥ 2^127`.
theorem trim_two_edge_lt_num (e : ℤ) : 2 * trimEdge e < trimNum e := by
  have hu : trimUnit e ≤ 2 ^ 68 := by
    rw [trimUnit]; exact Nat.pow_le_pow_right (by norm_num) (by omega)
  have hedge : trimEdge e ≤ 2 ^ 68 * trimDen e := by
    rw [trimEdge]; exact Nat.mul_le_mul_right _ hu
  have hnum := trim_num_lower e
  have hden := trim_den_pos e
  omega

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

-- One ULP of the value is narrower than one step of the coarse decimal grid:
-- `2·num < scale`. This depends on the low bits of the truncated power of ten,
-- not just its magnitude, so it is checked separately for each exponent.
theorem trim_two_num_lt_scale (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    2 * trimNum e < trimScale e := by
  have hcert := trim_narrow_all e (by simpa [Finset.mem_Icc] using he)
  simp only [trimNarrowHolds, decide_eq_true_eq] at hcert
  rw [← trim_sig_nat] at hcert
  have hstep : trimDen e * (2 * trimSig e + 2) ≤ trimDen e * trimModulus e :=
    Nat.mul_le_mul_left _ hcert
  have hexp : trimDen e * (2 * trimSig e + 2)
      = 2 * (trimDen e * trimSig e) + 2 * trimDen e := by ring
  have hsplit := trim_num_split e
  have hscale := trim_scale_split e
  have hmod : trimNum e % trimDen e < trimDen e := Nat.mod_lt _ (trim_den_pos e)
  omega

-- Whatever the step, the candidate scaled back up plus the gap is the scaled
-- value `2·f·num`: `Nat.div_add_mod` recovers the product from the quotient
-- and `trim_num_split` the power of ten from its truncation.
theorem step_quotient_add_gap (m f : ℕ) (e : ℤ) :
    2 * f * trimSig e / m * (m * trimDen e) + stepGap m f e
      = 2 * f * trimNum e := by
  rw [stepGap, stepResidue]
  calc 2 * f * trimSig e / m * (m * trimDen e)
        + (trimDen e * (2 * f * trimSig e % m)
          + 2 * f * (trimNum e % trimDen e))
      = trimDen e * (m * (2 * f * trimSig e / m) + 2 * f * trimSig e % m)
          + 2 * f * (trimNum e % trimDen e) := by ring
    _ = 2 * f * (trimDen e * trimSig e + trimNum e % trimDen e) := by
        rw [Nat.div_add_mod]; ring
    _ = 2 * f * trimNum e := by rw [trim_num_split]

-- The truncation error of the power of ten, `2·f·τ` with `τ < den` and
-- `2·f < 2^54`.
theorem trim_trunc_lt (f : ℕ) (e : ℤ) (h : Regular f e) :
    2 * f * (trimNum e % trimDen e) < 2 ^ 54 * trimDen e :=
  lt_of_le_of_lt (Nat.mul_le_mul_right _ (by have := h.2.1; omega))
    (mul_lt_mul_of_pos_left (Nat.mod_lt _ (trim_den_pos e)) (by positivity))

-- Whatever the step, the gap can overshoot it, but by less than `num`: the
-- residue stays below the step, and `num ≥ 2^127·den` absorbs the truncation
-- error.
theorem step_gap_lt_scale_add (m f : ℕ) (e : ℤ) (h : Regular f e)
    (hm : 0 < m) : stepGap m f e < m * trimDen e + trimNum e := by
  have hres : trimDen e * stepResidue m f e < m * trimDen e := by
    rw [stepResidue, Nat.mul_comm m]
    exact mul_lt_mul_of_pos_left (Nat.mod_lt _ hm) (trim_den_pos e)
  have hlow := trim_trunc_lt f e h
  have htrunc : 2 ^ 54 * trimDen e ≤ trimNum e :=
    le_trans
      (Nat.mul_le_mul_right _
        (Nat.pow_le_pow_right (by norm_num) (by norm_num)))
      (trim_num_lower e)
  rw [stepGap]
  omega

-- The bridge every trim bound crosses: the gap is `den` times the residue plus
-- the truncation error `2·f·τ`, and `trim_low_bits` keeps that error inside the
-- low bits of `p10` that the packed comparison discards anyway.
theorem trim_gap_sandwich (f : ℕ) (e : ℤ) (h : Regular f e) :
    trimDen e * trimResidue f e ≤ trimGap f e ∧
      trimGap f e ≤ trimDen e * trimResidue f e
        + trimSig e % trimUnit e * trimDen e := by
  rw [trimGap, stepGap, ← trimResidue]
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

-- The four windows the certificates refute, named by the boundary they hug and
-- the side they hug it from. Soundness uses the outward pair, completeness the
-- inward pair.
structure TrimGapSeparated (f : ℕ) (e : ℤ) : Prop where
  aboveNum : ¬(trimNum e < trimGap f e ∧ trimGap f e < trimBnd e)
  belowNum : ¬(trimNum e < trimGap f e + trimEdge e ∧ trimGap f e < trimNum e)
  belowScale : ¬(trimScale e ≤ trimGap f e + trimBnd e ∧
    trimGap f e + trimNum e < trimScale e)
  aboveScale : ¬(trimScale e < trimGap f e + trimNum e ∧
    trimGap f e + trimNum e < trimScale e + trimEdge e)

-- The certificates' semantic content: the gap never lands in those windows,
-- each of them narrower than one window edge. Below this theorem is modular
-- arithmetic of `2·num·f mod scale`; above it the certificates are hidden
-- behind these four separation facts.
theorem trim_gap_separated (f : ℕ) (e : ℤ) (h : Regular f e) :
    TrimGapSeparated f e := by
  obtain ⟨q, hcert⟩ := trim_windows_refuted e h.2.2
  simp only [trimCertificateValid, trimWindows, modWindowsRefuted,
    List.all_cons, List.all_nil, Bool.and_eq_true, and_true] at hcert
  obtain ⟨hc1, hc2, hc3, hc4⟩ := hcert
  have hbnd := trim_bnd_le_scale e (decimal_shift_lt_four e h.2.2)
  have hedge := trim_two_edge_lt_num e
  have hnarrow := trim_two_num_lt_scale e h.2.2
  refine ⟨?_, ?_, ?_, ?_⟩
  · rintro ⟨hlo, hhi⟩
    exact trim_no_window_hit f e h hc1 (by omega) (by omega) (by omega)
  · rintro ⟨hlo, hhi⟩
    exact trim_no_window_hit f e h hc2 (by omega) (by omega) (by omega)
  · rintro ⟨hlo, hhi⟩
    exact trim_no_window_hit f e h hc3 (by omega) (by omega) (by omega)
  · rintro ⟨hlo, hhi⟩
    exact trim_no_window_hit f e h hc4 (by omega) (by omega) (by omega)

-- Trim-down soundness: if the packed comparison reads `c ≤ halfUlp`, then
-- the exact gap is at most `num`; otherwise it would lie in a forbidden window.
theorem trim_gap_le (f : ℕ) (e : ℤ) (h : Regular f e)
    (hcmp : trimResidue f e / trimUnit e ≤ trimSig e / trimUnit e) :
    trimGap f e ≤ trimNum e := by
  by_contra! hcon
  exact (trim_gap_separated f e h).aboveNum ⟨hcon, trim_gap_box f e h hcmp⟩

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

-- Trim-down completeness: if the packed comparison reads `halfUlp < c`, then
-- the residue lies strictly above `p10`. After clearing the denominator that
-- single unit becomes a whole `den`, which exceeds the truncation remainder, so
-- `num < gap` without using a certificate.
theorem trim_num_lt_gap (f : ℕ) (e : ℤ) (h : Regular f e)
    (hcmp : trimSig e / trimUnit e < trimResidue f e / trimUnit e) :
    trimNum e < trimGap f e := by
  have hslack := add_mod_succ_le_of_div_lt (trim_unit_pos e) hcmp
  have hscaled : trimDen e * (trimSig e + 1) ≤ trimDen e * trimResidue f e :=
    Nat.mul_le_mul_left _ (by omega)
  have hexp : trimDen e * (trimSig e + 1)
      = trimDen e * trimSig e + trimDen e := by ring
  have hnum := trim_num_split e
  have hmod : trimNum e % trimDen e < trimDen e := Nat.mod_lt _ (trim_den_pos e)
  have hsand : trimDen e * trimResidue f e ≤ trimGap f e :=
    (trim_gap_sandwich f e h).1
  omega

-- Trim-down completeness, tie case: if the packed comparison has not read
-- `c < halfUlp`, then `num` is less than one window edge above `gap`.
-- Therefore a gap below `num` would lie in the forbidden inward window.
theorem trim_num_le_gap (f : ℕ) (e : ℤ) (h : Regular f e)
    (hcmp : trimSig e / trimUnit e ≤ trimResidue f e / trimUnit e) :
    trimNum e ≤ trimGap f e := by
  by_contra! hcon
  have hslack := add_mod_lt_of_div_le (trim_unit_pos e) hcmp
  have hscaled : trimDen e * (trimSig e + trimResidue f e % trimUnit e + 1)
      ≤ trimDen e * (trimResidue f e + trimUnit e) :=
    Nat.mul_le_mul_left _ (by omega)
  have hexp1 : trimDen e * (trimSig e + trimResidue f e % trimUnit e + 1)
      = trimDen e * trimSig e + trimDen e * (trimResidue f e % trimUnit e)
        + trimDen e := by ring
  have hexp2 : trimDen e * (trimResidue f e + trimUnit e)
      = trimDen e * trimResidue f e + trimEdge e := by
    rw [trimEdge]; ring
  have hnum := trim_num_split e
  have hmod : trimNum e % trimDen e < trimDen e := Nat.mod_lt _ (trim_den_pos e)
  have hsand : trimDen e * trimResidue f e ≤ trimGap f e :=
    (trim_gap_sandwich f e h).1
  have hnear : trimNum e < trimGap f e + trimEdge e := by omega
  exact (trim_gap_separated f e h).belowNum ⟨hnear, hcon⟩

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
  exact (trim_gap_separated f e h).belowScale ⟨hfar, hcon⟩

-- Equality in the packed comparison scales to
-- `gap + num = scale + (2f+1)·τ`: the truncation remainder is exactly the
-- slack in the corresponding exact bound.
theorem trim_tie_gap_eq (f : ℕ) (e : ℤ)
    (htie : trimModulus e = trimResidue f e + trimSig e) :
    trimGap f e + trimNum e
      = trimScale e + (2 * f + 1) * (trimNum e % trimDen e) := by
  let τ := trimNum e % trimDen e
  have hnum : trimNum e = trimDen e * trimSig e + τ := by
    simpa [τ] using (trim_num_split e).symm
  change trimDen e * trimResidue f e + 2 * f * τ + trimNum e =
    trimModulus e * trimDen e + (2 * f + 1) * τ
  rw [hnum, htie]
  ring

-- Packed equality forces `2^(129-h) ∣ p10`: since
-- `trimModulus = 5·2^(129-h)` and `2f+1` is odd, all of that power of two
-- must come from `p10`.
theorem trim_tie_pow_two_dvd (f : ℕ) (e : ℤ) (hsh : decimalShift e < 4)
    (htie : trimModulus e = trimResidue f e + trimSig e) :
    2 ^ (129 - decimalShift e) ∣ trimSig e := by
  have hw : trimResidue f e = 2 * f * trimSig e % trimModulus e := rfl
  have hdvd : trimModulus e ∣ (2 * f + 1) * trimSig e := by
    refine ⟨2 * f * trimSig e / trimModulus e + 1, ?_⟩
    have hq := Nat.div_add_mod (2 * f * trimSig e) (trimModulus e)
    calc (2 * f + 1) * trimSig e = 2 * f * trimSig e + trimSig e := by ring
      _ = trimModulus e * (2 * f * trimSig e / trimModulus e)
          + (trimResidue f e + trimSig e) := by rw [hw]; omega
      _ = trimModulus e * (2 * f * trimSig e / trimModulus e + 1) := by
          rw [← htie]; ring
  refine Nat.Coprime.dvd_of_dvd_mul_left
    (Nat.Coprime.pow_left _
      ((Nat.prime_two.coprime_iff_not_dvd).mpr (by omega)))
    (dvd_trans ⟨5, ?_⟩ hdvd)
  rw [trimModulus,
    show 129 - decimalShift e = (128 - decimalShift e) + 1 from by omega,
    pow_succ]
  ring

-- An exact cached power admits a tie only at `k = 0`. For `k > 0` its
-- denominator retains a factor of five; for `k < 0` it has too few factors of
-- two. At `k = 0`, `p10 = 2^127`, the special tie handled by `roundU0`.
theorem trim_exact_tie_k_zero (f : ℕ) (e : ℤ) (h : Regular f e)
    (hτ : trimNum e % trimDen e = 0)
    (htie : trimModulus e = trimResidue f e + trimSig e) :
    decimalExponent e = 0 := by
  have hsplit : trimDen e * trimSig e = trimNum e := by
    simpa [hτ] using trim_num_split e
  rcases lt_trichotomy (decimalExponent e) 0 with hk | hk | hk
  · exfalso
    have hsh := decimal_shift_lt_four e h.2.2
    have halign := align_all e (by simpa [Finset.mem_Icc] using h.2.2)
    -- Scaling by `log₁₀ 2 < 1` moves a negative exponent towards zero.
    have hek : e ≤ decimalExponent e := by
      unfold decimalExponent at hk ⊢
      omega
    have htie2 := trim_tie_pow_two_dvd f e hsh htie
    rw [trimNum, power10Num, trimDen, power10Den, neg_neg,
      show (decimalExponent e).toNat = 0 from by omega, pow_zero,
      one_mul] at hsplit
    set m := (-decimalExponent e).toNat
    set a := (128 - power10Exponent (-decimalExponent e)).toNat
    set b := (power10Exponent (-decimalExponent e) - 128).toNat
    -- The tie needs `129 - h` factors of two, but `5^m` supplies none.
    have hpow : 2 ^ (b + (129 - decimalShift e)) ∣ 10 ^ m * 2 ^ a := by
      rw [pow_add, ← hsplit]
      exact mul_dvd_mul_left (2 ^ b) htie2
    rw [show 10 ^ m * 2 ^ a = 5 ^ m * 2 ^ (m + a) by
      rw [show (10 : ℕ) = 5 * 2 from rfl, mul_pow, pow_add]
      ring] at hpow
    have hle := (Nat.pow_dvd_pow_iff_le_right (by norm_num : 1 < 2)).mp
      (Nat.Coprime.dvd_of_dvd_mul_left (Nat.Coprime.pow _ _ (by decide)) hpow)
    omega
  · exact hk
  · exfalso
    have h5den : 5 ∣ trimDen e := by
      rw [trimDen, power10Den, neg_neg]
      exact Dvd.dvd.mul_right
        (dvd_pow (by norm_num : (5 : ℕ) ∣ 10) (by omega)) _
    have h5num : ¬5 ∣ trimNum e := by
      rw [trimNum, power10Num,
        show (-decimalExponent e).toNat = 0 from by omega, pow_zero, one_mul]
      intro hcon
      have := Nat.prime_five.dvd_of_dvd_pow hcon
      norm_num at this
    exact h5num (dvd_trans h5den ⟨trimSig e, hsplit.symm⟩)

-- At `k = 0` the cached power of ten is exactly `2^127`, so it has no low bits
-- for the truncation to drop.
theorem trim_power_of_k_zero (e : ℤ) (hk : decimalExponent e = 0) :
    trimNum e = 2 ^ 127 * trimDen e ∧ trimSig e = 2 ^ 127 := by
  have heq : trimNum e = 2 ^ 127 * trimDen e := by
    rw [trimNum, trimDen, hk]
    decide
  exact ⟨heq, by rw [trim_sig_nat, heq, Nat.mul_div_cancel _ (trim_den_pos e)]⟩

-- At `k = 0` the packed comparison is exact: both operands are whole window
-- units and the cached-power truncation error vanishes. Thus the packed tie
-- lifts to an exact tie, with `gap + num = scale`.
theorem trim_gap_num_eq_scale_of_k_zero (f : ℕ) (e : ℤ) (h : Regular f e)
    (hk : decimalExponent e = 0)
    (htie : trimResidue f e / trimUnit e + trimSig e / trimUnit e
      = 10 * 2 ^ 60) :
    trimGap f e + trimNum e = trimScale e := by
  have hsh : decimalShift e < 4 := decimal_shift_lt_four e h.2.2
  obtain ⟨hnum, hsig⟩ := trim_power_of_k_zero e hk
  have hunit : trimUnit e ∣ trimSig e := by
    rw [hsig, trimUnit]
    exact pow_dvd_pow 2 (by omega)
  have hres : trimUnit e ∣ trimResidue f e := by
    refine (Nat.dvd_mod_iff ?_).mpr (hunit.mul_left _)
    rw [trimModulus, trimUnit]
    exact Dvd.dvd.mul_left (pow_dvd_pow 2 (by omega)) 10
  -- Whole units on both sides, so the packed tie is a tie of the operands.
  have hsum : trimResidue f e + trimSig e = trimModulus e := by
    rw [← Nat.mul_div_cancel' hres, ← Nat.mul_div_cancel' hunit, ← Nat.mul_add,
      htie]
    exact (trim_modulus_eq e hsh).symm
  have hgap : trimGap f e = trimDen e * trimResidue f e := by
    rw [trimGap, stepGap, ← trimResidue, hnum, Nat.mul_mod_left, Nat.mul_zero,
      Nat.add_zero]
  have hscaled_sum : trimDen e * (trimResidue f e + trimSig e)
      = trimDen e * trimResidue f e + trimNum e := by
    rw [hnum, hsig]; ring
  rw [hgap, ← trim_scale_split e, ← hsum, hscaled_sum]

-- Strict trim-up soundness for the final `t0 ≤ t1` test. Packed equality is
-- either made strict by cached-power truncation or is an exact tie, which
-- forces `k = 0` and belongs to the dedicated `k = 0 ∧ t1 = t0` branch.
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
  rcases lt_or_eq_of_le hcmp with hlt | heq
  · -- Clearing the denominator preserves the strict inequality.
    have hscaled : trimScale e
        < trimDen e * trimResidue f e + trimDen e * trimSig e := by
      rw [← trim_scale_split e, ← mul_add]
      exact mul_lt_mul_of_pos_left hlt (trim_den_pos e)
    have hnum : trimDen e * trimSig e + trimNum e % trimDen e = trimNum e :=
      trim_num_split e
    have hsand : trimDen e * trimResidue f e ≤ trimGap f e :=
      (trim_gap_sandwich f e h).1
    omega
  -- Equality in the packed comparison is a genuine tie only when the cached
  -- power is exact. Otherwise its truncation remainder is precisely the slack
  -- that makes the exact bound strict.
  · by_cases hτ : trimNum e % trimDen e = 0
    · -- A tie in exact arithmetic, which only `k = 0` admits.
      exfalso
      refine hne ⟨trim_exact_tie_k_zero f e h hτ heq, ?_⟩
      -- An exact tie is seen as one by the packed comparison too.
      have h4 : trimUnit e
          * (trimResidue f e / trimUnit e + trimSig e / trimUnit e)
          ≤ trimUnit e * (10 * 2 ^ 60) := by
        rw [← hmodeq, heq]; exact mul_div_add_div_le _ _ _
      exact Nat.le_antisymm (Nat.le_of_mul_le_mul_left h4 hu_pos) hb
    · rw [trim_tie_gap_eq f e heq]
      exact Nat.lt_add_of_pos_right
        (Nat.mul_pos (by omega) (Nat.pos_of_ne_zero hτ))

-- Each packed unit of room below the modulus becomes one `trimEdge` after
-- clearing the denominator. Reconstructing `gap + num` adds `(2f+1)·τ`, where
-- `τ` is the cached-power truncation remainder; `trim_high_bits` shows that
-- this error together with the discarded low bits of `p10` fits within one
-- `trimEdge`.
theorem trim_packed_room (f : ℕ) (e : ℤ) (h : Regular f e) (n : ℕ)
    (hcmp : trimResidue f e / trimUnit e + trimSig e / trimUnit e + n
      ≤ 10 * 2 ^ 60) :
    trimGap f e + trimNum e + n * trimEdge e
      ≤ trimScale e + trimResidue f e % trimUnit e * trimDen e
        + trimEdge e := by
  have hsh : decimalShift e < 4 := decimal_shift_lt_four e h.2.2
  -- Scaled back up, the packed sum leaves `n` whole window units of room.
  have hroom : trimResidue f e + trimSig e + n * trimUnit e
      ≤ trimModulus e + trimResidue f e % trimUnit e
        + trimSig e % trimUnit e := by
    have hscaled : trimUnit e * (trimResidue f e / trimUnit e
        + trimSig e / trimUnit e + n) ≤ trimUnit e * (10 * 2 ^ 60) :=
      Nat.mul_le_mul_left _ hcmp
    have hexp : trimUnit e * (trimResidue f e / trimUnit e
          + trimSig e / trimUnit e + n)
        = trimUnit e * (trimResidue f e / trimUnit e)
          + trimUnit e * (trimSig e / trimUnit e) + n * trimUnit e := by ring
    have hw := Nat.div_add_mod (trimResidue f e) (trimUnit e)
    have hp := Nat.div_add_mod (trimSig e) (trimUnit e)
    have hmod := trim_modulus_eq e hsh
    omega
  -- Clearing the denominator turns each unit of that room into one `trimEdge`.
  have hscaled : trimDen e * trimResidue f e
        + trimDen e * trimSig e + n * trimEdge e
      ≤ trimScale e + trimResidue f e % trimUnit e * trimDen e
        + trimSig e % trimUnit e * trimDen e := by
    have hs := Nat.mul_le_mul_left (trimDen e) hroom
    have hexp1 : trimDen e * (trimResidue f e + trimSig e + n * trimUnit e)
        = trimDen e * trimResidue f e + trimDen e * trimSig e
          + n * trimEdge e := by
      rw [trimEdge]; ring
    have hexp2 : trimDen e * (trimModulus e + trimResidue f e % trimUnit e
          + trimSig e % trimUnit e)
        = trimScale e + trimResidue f e % trimUnit e * trimDen e
          + trimSig e % trimUnit e * trimDen e := by
      rw [← trim_scale_split e]; ring
    rwa [hexp1, hexp2] at hs
  -- Reconstructing `gap + num` from those products adds `(2f+1)·τ`.
  have hsum : trimGap f e + trimNum e
      = trimDen e * trimResidue f e + trimDen e * trimSig e
        + (2 * f + 1) * (trimNum e % trimDen e) :=
    calc trimGap f e + trimNum e
        = trimDen e * trimResidue f e + 2 * f * (trimNum e % trimDen e)
            + (trimDen e * trimSig e + trimNum e % trimDen e) := by
          rw [trimGap, stepGap, ← trimResidue, trim_num_split e]
      _ = trimDen e * trimResidue f e + trimDen e * trimSig e
            + (2 * f + 1) * (trimNum e % trimDen e) := by ring
  have htau : (2 * f + 1) * (trimNum e % trimDen e)
      ≤ 2 ^ 54 * (trimNum e % trimDen e) :=
    Nat.mul_le_mul_right _ (by have := h.2.1; omega)
  have hhigh : 2 ^ 54 * (trimNum e % trimDen e)
      + trimSig e % trimUnit e * trimDen e ≤ trimEdge e := by
    rw [trimEdge]; exact trim_high_bits e h.2.2
  omega

-- The low bits of the residue are worth less than one window unit.
theorem trim_residue_low_lt_edge (f : ℕ) (e : ℤ) :
    trimResidue f e % trimUnit e * trimDen e < trimEdge e := by
  rw [trimEdge]
  exact mul_lt_mul_of_pos_right (Nat.mod_lt _ (trim_unit_pos e))
    (trim_den_pos e)

-- Trim-up completeness: if the packed sum is two window units short of the
-- modulus, one unit absorbs the discarded low bits and one remains, so
-- `gap + num` is strictly below `scale`. No certificate is needed.
theorem trim_gap_num_lt_scale (f : ℕ) (e : ℤ) (h : Regular f e)
    (hcmp : trimResidue f e / trimUnit e + trimSig e / trimUnit e + 2
      ≤ 10 * 2 ^ 60) :
    trimGap f e + trimNum e < trimScale e := by
  have hroom := trim_packed_room f e h 2 hcmp
  have hedge := trim_residue_low_lt_edge f e
  omega

-- Trim-up completeness, tie case: if the packed sum is one window unit short
-- of the modulus, `gap + num` cannot exceed `scale`. The truncated comparison
-- only bounds any excess to less than one window edge, and that inward window
-- is excluded by the certificate.
theorem trim_gap_num_le_scale (f : ℕ) (e : ℤ) (h : Regular f e)
    (hcmp : trimResidue f e / trimUnit e + trimSig e / trimUnit e + 1
      = 10 * 2 ^ 60) :
    trimGap f e + trimNum e ≤ trimScale e := by
  by_contra! hcon
  have hroom := trim_packed_room f e h 1 (by omega)
  have hedge := trim_residue_low_lt_edge f e
  exact (trim_gap_separated f e h).aboveScale ⟨hcon, by omega⟩

-- The free side of the trim-up bound, at the multiple-of-ten step.
theorem trim_gap_lt_scale_add (f : ℕ) (e : ℤ) (h : Regular f e) :
    trimGap f e < trimScale e + trimNum e :=
  step_gap_lt_scale_add _ f e h (by rw [trimModulus]; positivity)

/-! ### From integer bounds to half-ULP bounds

In the scale `trimMul` the two trimmed candidates have scaled errors `-trimGap`
and `trimScale - trimGap`, and half a scaled ULP is exactly `trimNum`. The
power of ten enters only through `trim_mul_eq`, which expresses `trimMul` as
`trimNum` times the inverse scale `s = 2^(1-e)·10^k`. Thus each candidate
bound `|cand - x| ≤ u/2` is a comparison of `trimGap` with `trimNum`.
-/

-- The unit step, cleared. The window step is ten of them.
def trimMul (e : ℤ) : ℕ := 2 ^ (128 - decimalShift e) * trimDen e

theorem trim_scale_eq_ten_mul (e : ℤ) : trimScale e = 10 * trimMul e := by
  simp only [trimScale, trimMul, trimModulus]
  ring

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

-- The scale sends half a scaled ULP to `trimNum`.
theorem trim_mul_half_ulp (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
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

-- The scale sends the scaled value to `2·f·num`.
theorem trim_mul_value (f : ℕ) (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    value f e * (10 ^ decimalExponent e)⁻¹ * (trimMul e : ℚ)
      = 2 * f * (trimNum e : ℚ) := by
  rw [← trim_mul_half_ulp e he, value, ulp]
  ring

-- Both candidates reach ℚ the same way: cleared of the scale, a candidate
-- accounting for the whole product except its gap sits exactly that gap below
-- the scaled value.
theorem scaled_error_of_nat {cand gap : ℕ} (f : ℕ) (e : ℤ)
    (he : -1074 ≤ e ∧ e ≤ 971)
    (hnat : cand * trimMul e + gap = 2 * f * trimNum e) :
    ((cand : ℚ) - value f e * (10 ^ decimalExponent e)⁻¹) * (trimMul e : ℚ)
      = -(gap : ℚ) := by
  have hcast : (cand : ℚ) * trimMul e + gap = 2 * f * trimNum e := by
    exact_mod_cast hnat
  linear_combination hcast - trim_mul_value f e he

-- The trim-down candidate sits exactly `trimGap` below the scaled value: it is
-- the quotient at the window step, which is ten unit steps.
theorem dec_ten_down_scaled_error (f : ℕ) (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    (((sigHi f e - sigHi f e % 10 : ℕ) : ℚ)
        - value f e * (10 ^ decimalExponent e)⁻¹) * (trimMul e : ℚ)
      = -(trimGap f e : ℚ) :=
  scaled_error_of_nat f e he <| by
    calc (sigHi f e - sigHi f e % 10) * trimMul e + trimGap f e
        = 2 * f * trimSig e / trimModulus e * trimScale e + trimGap f e := by
          rw [sig_hi_ten_quotient f e (decimal_shift_lt_four e he),
            trim_scale_eq_ten_mul]
          ring
      _ = 2 * f * trimNum e := step_quotient_add_gap _ f e

-- The trim-up candidate sits `trimScale - trimGap` above the scaled value,
-- since the scale sends a decimal step of `10` to the window modulus.
theorem dec_ten_up_scaled_error (f : ℕ) (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    (((sigHi f e - sigHi f e % 10 : ℕ) : ℚ) + 10
        - value f e * (10 ^ decimalExponent e)⁻¹) * (trimMul e : ℚ)
      = (trimScale e : ℚ) - (trimGap f e : ℚ) := by
  have hten : (10 : ℚ) * (trimMul e : ℚ) = (trimScale e : ℚ) := by
    exact_mod_cast (trim_scale_eq_ten_mul e).symm
  linear_combination dec_ten_down_scaled_error f e he + hten

-- Scaling by `trimMul` loses nothing: a candidate with scaled error `dist` is
-- within half a ULP exactly when `|dist|` is at most `trimNum`.
theorem half_ulp_iff_scaled_error {cand dist : ℚ} (f : ℕ) (e : ℤ)
    (he : -1074 ≤ e ∧ e ≤ 971)
    (hscale : (cand - value f e * (10 ^ decimalExponent e)⁻¹) * (trimMul e : ℚ)
      = dist) :
    let k := decimalExponent e
    let x := value f e * (10 ^ k)⁻¹
    let u := ulp e * (10 ^ k)⁻¹
    (|cand - x| ≤ u / 2 ↔ |dist| ≤ (trimNum e : ℚ)) ∧
      (|cand - x| < u / 2 ↔ |dist| < (trimNum e : ℚ)) := by
  intro k x u
  have hpos : (0 : ℚ) < (trimMul e : ℚ) :=
    Nat.cast_pos.mpr
      (by rw [trimMul]; exact Nat.mul_pos (by positivity) (trim_den_pos e))
  have hdist : |cand - x| * (trimMul e : ℚ) = |dist| := by
    rw [← hscale, abs_mul, abs_of_pos hpos]
  exact ⟨by rw [← mul_le_mul_iff_of_pos_right hpos, hdist,
      trim_mul_half_ulp e he],
    by rw [← mul_lt_mul_iff_of_pos_right hpos, hdist, trim_mul_half_ulp e he]⟩

-- Scaling by the positive factor `10^k` preserves the rounding bounds, so
-- round-tripping is equivalent to the significand bound in candidate units.
theorem roundtrips_iff_scaled (f : ℕ) (e : ℤ) (d : ℕ) :
    let k := decimalExponent e
    let x := value f e * (10 ^ k)⁻¹
    let u := ulp e * (10 ^ k)⁻¹
    Roundtrips f e (d * 10 ^ k)
      ↔ (if f % 2 = 0 then |(d : ℚ) - x| ≤ u / 2 else |(d : ℚ) - x| < u / 2) := by
  intro k x u
  have hp : (0 : ℚ) < 10 ^ k := by positivity
  have hne : (10 : ℚ) ^ k ≠ 0 := ne_of_gt hp
  have hdist : |(d : ℚ) - x| * 10 ^ k = |(d : ℚ) * 10 ^ k - value f e| := by
    have h : ((d : ℚ) - x) * 10 ^ k = (d : ℚ) * 10 ^ k - value f e := by
      simp only [x]; field_simp
    rw [← h, abs_mul, abs_of_pos hp]
  have hhalf : u / 2 * 10 ^ k = ulp e / 2 := by
    simp only [u]; field_simp
  simp only [Roundtrips]
  split_ifs
  · rw [← hdist, ← hhalf]; exact mul_le_mul_iff_of_pos_right hp
  · rw [← hdist, ← hhalf]; exact mul_lt_mul_iff_of_pos_right hp

/-! ### The unit-step candidate

`decOne` is `sigHi` rounded to nearest using the discarded word `sigLo`. In
the scale `trimMul = 2^(128-h)·den`, `sigHi` sits `oneGap` below the scaled
value and rounding up adds one whole `trimMul`. The `roundU1` test bounds the
remainder relative to half the window `2^(128-h)`, with only the bits below
`sigLo` unseen. The remaining margin follows from `power10_margin_all`; at
`k = 0` the candidate is exact instead.
-/

-- The same residue and gap at the unit step: `sigHi·2^(128-h) + oneResidue`
-- is the product `2·f·p10`, and `oneGap` is the distance from `sigHi` to the
-- scaled value once the denominator is cleared.
def oneResidue (f : ℕ) (e : ℤ) : ℕ :=
  stepResidue (2 ^ (128 - decimalShift e)) f e

def oneGap (f : ℕ) (e : ℤ) : ℕ := stepGap (2 ^ (128 - decimalShift e)) f e

-- `sigHi` scaled up, plus the gap, is the scaled value `2·f·num`.
theorem sig_hi_add_one_gap (f : ℕ) (e : ℤ) (hsh : decimalShift e < 4) :
    sigHi f e * trimMul e + oneGap f e = 2 * f * trimNum e := by
  rw [sig_hi_quotient f e hsh]
  exact step_quotient_add_gap _ f e

-- What `roundU1` says about the remainder: yy compares the discarded word with
-- half its range, so the test is on the remainder against half the window
-- `2^(128-h)`, blind only to the `2^(64-h)` bits below `sigLo`.
theorem one_round_half (f : ℕ) (e : ℤ) (hsh : decimalShift e < 4) :
    ((toDecimalCandidates f e).roundU1 = true →
        2 ^ (127 - decimalShift e) ≤ oneResidue f e) ∧
      ((toDecimalCandidates f e).roundU1 = false →
        oneResidue f e
          < 2 ^ (127 - decimalShift e) + 2 ^ (64 - decimalShift e)) := by
  have hpos : (0 : ℕ) < 2 ^ (64 - decimalShift e) := by positivity
  -- `sigLo` is the remainder with the bits below the last kept one dropped.
  have hlo : sigLo f e = oneResidue f e / 2 ^ (64 - decimalShift e) := by
    rw [sig_lo_eq, oneResidue, stepResidue, pow_shift_split e 128 (by omega),
      Nat.mul_mod_mul_left, pow_shift_split e 64 (by omega),
      Nat.mul_div_mul_left _ _ (by positivity)]
  have hpow : (2 : ℕ) ^ 63 * 2 ^ (64 - decimalShift e)
      = 2 ^ (127 - decimalShift e) := by
    rw [← pow_add]
    congr 1
    omega
  have hround : (toDecimalCandidates f e).roundU1
      = if sigLo f e = 2 ^ 63 then decide (sigHi f e % 2 = 1)
        else decide (2 ^ 63 < sigLo f e) := rfl
  constructor
  · intro hu1
    have h63 : 2 ^ 63 ≤ sigLo f e := by
      rw [hround] at hu1
      split at hu1
      · rename_i heq; exact heq.ge
      · exact (of_decide_eq_true hu1).le
    rwa [hlo, Nat.le_div_iff_mul_le hpos, hpow] at h63
  · intro hu1
    have h63 : sigLo f e ≤ 2 ^ 63 := by
      rw [hround] at hu1
      split at hu1
      · rename_i heq; exact heq.le
      · exact not_lt.mp (of_decide_eq_false hu1)
    rw [hlo] at h63
    calc oneResidue f e
        < (2 ^ 63 + 1) * 2 ^ (64 - decimalShift e) :=
          (Nat.div_lt_iff_lt_mul hpos).mp (Nat.lt_succ_of_le h63)
      _ = 2 ^ (127 - decimalShift e) + 2 ^ (64 - decimalShift e) := by
          rw [add_mul, one_mul, hpow]

-- At `k = 0` the power of ten is exactly `2^127`, so `2·f·p10` fills whole
-- windows and `sigHi` is already the scaled value.
theorem one_exact_of_k_zero (f : ℕ) (e : ℤ) (hsh : decimalShift e < 4)
    (hk : decimalExponent e = 0) :
    oneResidue f e = 0 ∧ oneGap f e = 0 := by
  obtain ⟨heq, hsig⟩ := trim_power_of_k_zero e hk
  have hres : oneResidue f e = 0 := by
    rw [oneResidue, stepResidue, hsig,
      show 2 * f * 2 ^ 127
          = f * 2 ^ decimalShift e * 2 ^ (128 - decimalShift e) from by
        calc 2 * f * 2 ^ 127 = f * 2 ^ 128 := by ring
          _ = f * (2 ^ decimalShift e * 2 ^ (128 - decimalShift e)) := by
            rw [pow_shift_split e 128 (by omega)]
          _ = f * 2 ^ decimalShift e * 2 ^ (128 - decimalShift e) := by ring]
    exact Nat.mul_mod_left _ _
  refine ⟨hres, ?_⟩
  rw [oneGap, stepGap, ← oneResidue, hres, heq, Nat.mul_mod_left]
  simp

-- The certificate in the quantities the trim layer uses.
theorem one_margin (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971)
    (hk : decimalExponent e ≠ 0) :
    (2 ^ 127 + 2 ^ 65) * trimDen e ≤ trimNum e := by
  have hrange : decimalExponent e ∈ Finset.Icc (-324 : ℤ) 292 := by
    simp only [Finset.mem_Icc]
    unfold decimalExponent
    omega
  rw [trimNum, trimDen]
  exact (power10_margin_all _ hrange).resolve_left hk

-- Half a window, plus the bits `sigLo` cannot see and the truncation error of
-- the power of ten, still fits inside `num`. The half window is at most
-- `2^127·den` whatever the shift, so this is the certificate with room to
-- spare.
theorem one_half_window_lt (f : ℕ) (e : ℤ) (h : Regular f e)
    (hk : decimalExponent e ≠ 0) :
    trimDen e * (2 ^ (127 - decimalShift e) + 2 ^ (64 - decimalShift e))
        + 2 * f * (trimNum e % trimDen e) < trimNum e := by
  have hden := trim_den_pos e
  have hlow := trim_trunc_lt f e h
  calc trimDen e * (2 ^ (127 - decimalShift e) + 2 ^ (64 - decimalShift e))
          + 2 * f * (trimNum e % trimDen e)
      < trimDen e * (2 ^ 127 + 2 ^ 64) + 2 ^ 54 * trimDen e :=
        Nat.add_lt_add_of_le_of_lt
          (Nat.mul_le_mul_left _ (Nat.add_le_add
            (Nat.pow_le_pow_right (by norm_num) (by omega))
            (Nat.pow_le_pow_right (by norm_num) (by omega)))) hlow
    _ = (2 ^ 127 + 2 ^ 64 + 2 ^ 54) * trimDen e := by ring
    _ < (2 ^ 127 + 2 ^ 65) * trimDen e :=
        mul_lt_mul_of_pos_right (by norm_num) hden
    _ ≤ trimNum e := one_margin e h.2.2 hk

-- Rounding down leaves the whole gap inside `num`.
theorem one_gap_lt (f : ℕ) (e : ℤ) (h : Regular f e)
    (hu1 : (toDecimalCandidates f e).roundU1 = false) :
    oneGap f e < trimNum e := by
  have hsh := decimal_shift_lt_four e h.2.2
  by_cases hk : decimalExponent e = 0
  · rw [(one_exact_of_k_zero f e hsh hk).2]
    exact lt_of_lt_of_le (Nat.mul_pos (by positivity) (trim_den_pos e))
      (trim_num_lower e)
  · calc oneGap f e
        ≤ trimDen e * (2 ^ (127 - decimalShift e) + 2 ^ (64 - decimalShift e))
            + 2 * f * (trimNum e % trimDen e) :=
          Nat.add_le_add_right
            (Nat.mul_le_mul_left _ ((one_round_half f e hsh).2 hu1).le) _
      _ < trimNum e := one_half_window_lt f e h hk

-- Rounding up overshoots by less than `num`, and the gap it jumped is itself
-- less than `trimMul` past `num`.
theorem one_scale_bounds (f : ℕ) (e : ℤ) (h : Regular f e)
    (hu1 : (toDecimalCandidates f e).roundU1 = true) :
    trimMul e < oneGap f e + trimNum e ∧ oneGap f e < trimMul e + trimNum e := by
  have hsh := decimal_shift_lt_four e h.2.2
  have hres := (one_round_half f e hsh).1 hu1
  -- `k` is not zero, or the remainder would be zero and `roundU1` could not
  -- have fired.
  have hk : decimalExponent e ≠ 0 := by
    intro hk0
    rw [(one_exact_of_k_zero f e hsh hk0).1] at hres
    exact absurd hres (Nat.not_le.mpr (by positivity))
  have hmargin := one_half_window_lt f e h hk
  have hhalf : trimDen e * 2 ^ (127 - decimalShift e) < trimNum e :=
    lt_of_le_of_lt (le_trans (Nat.mul_le_mul_left _ (Nat.le_add_right _ _))
      (Nat.le_add_right _ _)) hmargin
  refine ⟨?_, step_gap_lt_scale_add _ f e h (by positivity)⟩
  calc trimMul e
      = trimDen e * 2 ^ (127 - decimalShift e)
          + trimDen e * 2 ^ (127 - decimalShift e) := by
        rw [trimMul,
          show (2 : ℕ) ^ (128 - decimalShift e)
              = 2 * 2 ^ (127 - decimalShift e) from by
            rw [← pow_succ']; congr 1; omega]
        ring
    _ < trimNum e + oneGap f e :=
        Nat.add_lt_add_of_lt_of_le hhalf
          (le_trans (Nat.mul_le_mul_left _ hres) (Nat.le_add_right _ _))
    _ = oneGap f e + trimNum e := Nat.add_comm _ _

-- `sigHi` sits exactly `oneGap` below the scaled value.
theorem sig_hi_scaled_error (f : ℕ) (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    ((sigHi f e : ℚ) - value f e * (10 ^ decimalExponent e)⁻¹)
        * (trimMul e : ℚ)
      = -(oneGap f e : ℚ) :=
  scaled_error_of_nat f e he
    (sig_hi_add_one_gap f e (decimal_shift_lt_four e he))

-- The unit-step candidate is strictly within half a scaled ULP of the exact
-- value.
theorem dec_one_error_bound (f : ℕ) (e : ℤ) (h : Regular f e) :
    let c := toDecimalCandidates f e
    let x := value f e * (10 ^ c.k)⁻¹
    let u := ulp e * (10 ^ c.k)⁻¹
    |(c.decOne : ℚ) - x| < u / 2 := by
  intro c x u
  have he := h.2.2
  by_cases hu1 : c.roundU1 = true
  -- Rounding up: the candidate is `trimMul - oneGap` above the value.
  · have hdec : ((c.decOne : ℕ) : ℚ) = (sigHi f e : ℚ) + 1 := by
      show ((sigHi f e + if c.roundU1 then 1 else 0 : ℕ) : ℚ) = _
      simp [hu1]
    obtain ⟨hbelow, habove⟩ := one_scale_bounds f e h hu1
    obtain ⟨_, hlt⟩ := half_ulp_iff_scaled_error f e he
      (cand := (sigHi f e : ℚ) + 1)
      (dist := (trimMul e : ℚ) - (oneGap f e : ℚ))
      (by linear_combination sig_hi_scaled_error f e he)
    rw [hdec]
    refine hlt.mpr (abs_lt.mpr ⟨?_, ?_⟩)
    · have : (oneGap f e : ℚ) < (trimMul e : ℚ) + (trimNum e : ℚ) := by
        exact_mod_cast habove
      linarith
    · have : (trimMul e : ℚ) < (oneGap f e : ℚ) + (trimNum e : ℚ) := by
        exact_mod_cast hbelow
      linarith
  -- Rounding down: the candidate is `oneGap` below it.
  · rw [Bool.not_eq_true] at hu1
    have hdec : ((c.decOne : ℕ) : ℚ) = (sigHi f e : ℚ) := by
      show ((sigHi f e + if c.roundU1 then 1 else 0 : ℕ) : ℚ) = _
      simp [hu1]
    obtain ⟨_, hlt⟩ :=
      half_ulp_iff_scaled_error f e he (sig_hi_scaled_error f e he)
    rw [hdec]
    refine hlt.mpr ?_
    rw [abs_neg, abs_of_nonneg (Nat.cast_nonneg _)]
    exact_mod_cast one_gap_lt f e h hu1

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
  have he := h.2.2
  have hsh : decimalShift e < 4 := decimal_shift_lt_four e he
  obtain ⟨hle, hlt⟩ :=
    half_ulp_iff_scaled_error f e he (dec_ten_down_scaled_error f e he)
  have habs : |(-(trimGap f e : ℚ))| = (trimGap f e : ℚ) := by
    rw [abs_neg]; exact abs_of_nonneg (Nat.cast_nonneg _)
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
    exact hle.mpr (by
      rw [habs]
      exact_mod_cast trim_gap_le f e h htie.symm.le)
  -- The strict comparison `c < halfUlp`.
  · have hd := hlt.mpr (by
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
  have he := h.2.2
  have hsh : decimalShift e < 4 := decimal_shift_lt_four e he
  obtain ⟨hle, hlt⟩ :=
    half_ulp_iff_scaled_error f e he (dec_ten_up_scaled_error f e he)
  -- The free side, shared by all branches.
  have hfree : -(trimNum e : ℚ) < (trimScale e : ℚ) - (trimGap f e : ℚ) := by
    have hz : (trimGap f e : ℚ) < (trimScale e : ℚ) + (trimNum e : ℚ) := by
      exact_mod_cast trim_gap_lt_scale_add f e h
    linarith
  -- Convert the integer bounds to scaled error bounds.
  have hle_of_pack (hpack : trimScale e ≤ trimGap f e + trimNum e) :
      |(ten : ℚ) + 10 - x| ≤ u / 2 := by
    refine hle.mpr (abs_le.mpr ⟨hfree.le, ?_⟩)
    have hz : (trimScale e : ℚ) ≤ (trimGap f e : ℚ) + (trimNum e : ℚ) := by
      exact_mod_cast hpack
    linarith
  have hlt_of_pack (hpack : trimScale e < trimGap f e + trimNum e) :
      |(ten : ℚ) + 10 - x| < u / 2 := by
    refine hlt.mpr (abs_lt.mpr ⟨hfree, ?_⟩)
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

/-! ### Round-trip correctness -/

-- The decimal significand produced by yy is within half a scaled ULP
-- of the exact value, with equality allowed only when `f` is even.
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
  have hk : k = decimalExponent e := by
    have h2 : (toDecimal f e).2 = decimalExponent e := rfl
    rwa [hdk] at h2
  subst hk
  exact (roundtrips_iff_scaled f e d).mpr
    (by simpa [hdk, zpow_neg] using decimal_significand_error_bound f e h)

/-! ### Shortness

`decOne` lies on yy's unit decimal grid, while the two trim candidates lie on
the grid one decimal digit coarser. yy emits `decOne` exactly when neither trim
flag fires, and it could have dropped a digit exactly when the rounding
interval contains a multiple of ten.

yy represents a dropped digit by leaving a trailing zero in the significand
rather than incrementing the exponent. Its result is therefore reduced before
stating full shortness.

The rounding interval is narrower than one coarse step, so the two trim
candidates are the only multiples of ten it can contain. `dec_ten_down` and
`dec_ten_up` prove that a firing trim flag produces a round-tripping candidate;
`round_d0_of_ten_down_roundtrips` and `round_u0_of_ten_up_roundtrips` prove the
converses.

For even `f` every tie branch accepts, so a missing flag leaves the packed
comparison a whole window unit clear of its boundary, enough to dominate the
cached-power truncation error. For odd `f` the tie branches reject, so a
missing flag at a packed tie can leave the exact gap within one window unit on
the rejected side of the boundary. The corresponding inward windows are
excluded by `trim_gap_separated`. The exceptional `roundU0` tie at `k = 0`
lies one unit farther out; there the cached power of ten is exact, and
`trim_gap_num_eq_scale_of_k_zero` settles it directly.
-/

-- Round-tripping in the candidate scale: the scaled candidate is within `num`
-- of the scaled value, with the strict odd case weakened to `≤`.
theorem roundtrips_bound (f : ℕ) (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) (d : ℕ)
    (hr : Roundtrips f e (d * 10 ^ decimalExponent e)) :
    |(d : ℚ) * (trimMul e : ℚ) - 2 * f * (trimNum e : ℚ)| ≤ (trimNum e : ℚ) := by
  have hscale :
      ((d : ℚ) - value f e * (10 ^ decimalExponent e)⁻¹) * (trimMul e : ℚ)
        = (d : ℚ) * (trimMul e : ℚ) - 2 * f * (trimNum e : ℚ) := by
    rw [sub_mul, trim_mul_value f e he]
  obtain ⟨hle, _⟩ := half_ulp_iff_scaled_error f e he hscale
  refine hle.mp ?_
  have hs := (roundtrips_iff_scaled f e d).mp hr
  by_cases hev : f % 2 = 0
  · simpa [hev] using hs
  · exact le_of_lt (by simpa [hev] using hs)

-- The candidate scale factor is positive.
theorem trim_mul_pos (e : ℤ) : (0 : ℚ) < (trimMul e : ℚ) :=
  Nat.cast_pos.mpr
    (by rw [trimMul]; exact Nat.mul_pos (by positivity) (trim_den_pos e))

-- A round-tripping multiple of ten is one of the two yy considers. The
-- round-trip interval is narrower than one coarse step, and its position
-- relative to the trim-down candidate excludes every other multiple of ten.
theorem coarse_candidate_cases (f : ℕ) (e : ℤ) (h : Regular f e) (d : ℕ)
    (h10 : d % 10 = 0)
    (hr : Roundtrips f e (d * 10 ^ decimalExponent e)) :
    d = sigHi f e - sigHi f e % 10 ∨ d = sigHi f e - sigHi f e % 10 + 10 := by
  let ten := sigHi f e - sigHi f e % 10
  have he := h.2.2
  have hmul := trim_mul_pos e
  obtain ⟨hlo, hhi⟩ := abs_le.mp (roundtrips_bound f e he d hr)
  -- Against the trim-down candidate, `d` is that error plus the gap away.
  have hdiff : ((d : ℚ) - (ten : ℚ)) * (trimMul e : ℚ)
      = ((d : ℚ) * (trimMul e : ℚ) - 2 * f * (trimNum e : ℚ))
        + (trimGap f e : ℚ) := by
    simp only [ten]
    linear_combination -trim_mul_value f e he - dec_ten_down_scaled_error f e he
  have hgap0 : (0 : ℚ) ≤ (trimGap f e : ℚ) := by positivity
  have hgap : (trimGap f e : ℚ) < (trimScale e : ℚ) + (trimNum e : ℚ) := by
    exact_mod_cast trim_gap_lt_scale_add f e h
  have hnarrow : 2 * (trimNum e : ℚ) < (trimScale e : ℚ) := by
    exact_mod_cast trim_two_num_lt_scale e he
  have hstep : (trimScale e : ℚ) = 10 * (trimMul e : ℚ) := by
    exact_mod_cast trim_scale_eq_ten_mul e
  -- Half a coarse step below the lower candidate, two steps above it.
  have hdown : (-5 : ℚ) * (trimMul e : ℚ)
      < ((d : ℚ) - (ten : ℚ)) * (trimMul e : ℚ) := by
    rw [hdiff]; linarith
  have hup : ((d : ℚ) - (ten : ℚ)) * (trimMul e : ℚ) < 20 * (trimMul e : ℚ) := by
    rw [hdiff]; linarith
  have h5 : ten < d + 5 := by
    exact_mod_cast (show (ten : ℚ) < (d : ℚ) + 5 by
      linarith [lt_of_mul_lt_mul_right hdown hmul.le])
  have h20 : d < ten + 20 := by
    exact_mod_cast (show (d : ℚ) < (ten : ℚ) + 20 by
      linarith [lt_of_mul_lt_mul_right hup hmul.le])
  omega

-- Two distinct multiples of ten are at least one coarse step apart, while the
-- rounding interval is narrower than one coarse step, so at most one can
-- round-trip.
theorem coarse_roundtrip_unique (f : ℕ) (e : ℤ) (h : Regular f e) (c₁ c₂ : ℕ)
    (h₁ : c₁ % 10 = 0) (h₂ : c₂ % 10 = 0)
    (hr₁ : Roundtrips f e (c₁ * 10 ^ decimalExponent e))
    (hr₂ : Roundtrips f e (c₂ * 10 ^ decimalExponent e)) :
    c₁ = c₂ := by
  have he := h.2.2
  have hmul := trim_mul_pos e
  obtain ⟨hlo₁, hhi₁⟩ := abs_le.mp (roundtrips_bound f e he c₁ hr₁)
  obtain ⟨hlo₂, hhi₂⟩ := abs_le.mp (roundtrips_bound f e he c₂ hr₂)
  have hnarrow : 2 * (trimNum e : ℚ) < (trimScale e : ℚ) := by
    exact_mod_cast trim_two_num_lt_scale e he
  have hstep : (trimScale e : ℚ) = 10 * (trimMul e : ℚ) := by
    exact_mod_cast trim_scale_eq_ten_mul e
  have hc₁ : c₁ < c₂ + 10 := by
    have hq := lt_of_mul_lt_mul_right (show ((c₁ : ℚ) - c₂) * (trimMul e : ℚ)
      < 10 * (trimMul e : ℚ) by linarith) hmul.le
    exact_mod_cast (show (c₁ : ℚ) < (c₂ : ℚ) + 10 by linarith)
  have hc₂ : c₂ < c₁ + 10 := by
    have hq := lt_of_mul_lt_mul_right (show ((c₂ : ℚ) - c₁) * (trimMul e : ℚ)
      < 10 * (trimMul e : ℚ) by linarith) hmul.le
    exact_mod_cast (show (c₂ : ℚ) < (c₁ : ℚ) + 10 by linarith)
  omega

-- A value in the `Regular` range is more than half a ULP away from zero, so
-- the zero significand never round-trips.
theorem not_roundtrips_zero (f : ℕ) (e : ℤ) (h : Regular f e) :
    ¬Roundtrips f e 0 := by
  have hf : (2 : ℚ) ^ 52 < (f : ℚ) := by exact_mod_cast h.1
  have hone : (1 : ℚ) ≤ 2 ^ 52 := by norm_num
  have hpos : (0 : ℚ) < 2 ^ e := by positivity
  have hval : |(0 : ℚ) - value f e| = (f : ℚ) * 2 ^ e := by
    rw [zero_sub, abs_neg, value, abs_of_pos (mul_pos (by linarith) hpos)]
  have hbig : ulp e / 2 < (f : ℚ) * 2 ^ e := by
    rw [ulp]
    nlinarith
  simp only [Roundtrips, hval]
  split_ifs <;> linarith

-- Completeness for the trim-down candidate: if it round-trips, `roundD0`
-- fires.
theorem round_d0_of_ten_down_roundtrips (f : ℕ) (e : ℤ) (h : Regular f e)
    (hr : Roundtrips f e
      ((sigHi f e - sigHi f e % 10 : ℕ) * 10 ^ decimalExponent e)) :
    (toDecimalCandidates f e).roundD0 = true := by
  have he := h.2.2
  have hsh : decimalShift e < 4 := decimal_shift_lt_four e he
  obtain ⟨hle, hlt⟩ :=
    half_ulp_iff_scaled_error f e he (dec_ten_down_scaled_error f e he)
  have hs := (roundtrips_iff_scaled f e _).mp hr
  -- yy's packed operands corresponding to `c` and `halfUlp`.
  set w := trimResidue f e / trimUnit e with hw
  set p := trimSig e / trimUnit e with hp
  -- The two yy tests, in the window quantities they actually compare.
  have hd0 : (toDecimalCandidates f e).roundD0
      = if p = w then decide (f % 2 = 0) else decide (w < p) := by
    rw [hw, hp, ← trim_c_eq f e hsh, ← trim_half_ulp_eq e hsh]
    rfl
  rw [hd0]
  by_cases heven : f % 2 = 0
  -- Even `f`: round-tripping gives `gap ≤ num`, so `w` cannot have passed `p`,
  -- which would put the gap a whole unit beyond `num`.
  · simp only [heven, reduceIte] at hs
    have hgap : trimGap f e ≤ trimNum e := by
      have hq := hle.mp hs
      rw [abs_neg, abs_of_nonneg (Nat.cast_nonneg _)] at hq
      exact_mod_cast hq
    have hcmp : w ≤ p := by
      by_contra hcon
      exact absurd (trim_num_lt_gap f e h (by omega)) (by omega)
    split
    · simpa using heven
    · simp only [decide_eq_true_eq]
      omega
  -- Odd `f`: round-tripping gives `gap < num`, which excludes the tie and
  -- forces the strict comparison.
  · simp only [heven, reduceIte] at hs
    have hgap : trimGap f e < trimNum e := by
      have hq := hlt.mp hs
      rw [abs_neg, abs_of_nonneg (Nat.cast_nonneg _)] at hq
      exact_mod_cast hq
    have hcmp : w < p := by
      by_contra hcon
      exact absurd (trim_num_le_gap f e h (by omega)) (by omega)
    split
    · omega
    · simpa using hcmp

-- Completeness for the trim-up candidate: if it round-trips, `roundU0` fires.
theorem round_u0_of_ten_up_roundtrips (f : ℕ) (e : ℤ) (h : Regular f e)
    (hr : Roundtrips f e
      ((sigHi f e - sigHi f e % 10 + 10 : ℕ) * 10 ^ decimalExponent e)) :
    (toDecimalCandidates f e).roundU0 = true := by
  have he := h.2.2
  have hsh : decimalShift e < 4 := decimal_shift_lt_four e he
  obtain ⟨hle, hlt⟩ :=
    half_ulp_iff_scaled_error f e he (dec_ten_up_scaled_error f e he)
  have hcast : ((sigHi f e - sigHi f e % 10 + 10 : ℕ) : ℚ)
      = ((sigHi f e - sigHi f e % 10 : ℕ) : ℚ) + 10 := by push_cast; ring
  have hs := (roundtrips_iff_scaled f e _).mp hr
  rw [hcast] at hs
  -- yy's two packed operands, `t1 = w + p` against the modulus `t0 = 10·2^60`.
  set w := trimResidue f e / trimUnit e with hw
  set p := trimSig e / trimUnit e with hp
  -- The three yy tests, in the window quantities they actually compare.
  have hu0 : (toDecimalCandidates f e).roundU0
      = (if w + p + 1 = 10 * 2 ^ 60 then
          decide (f % 2 = 0)
        else if decimalExponent e = 0 ∧ w + p = 10 * 2 ^ 60 then
          decide (f % 2 = 0)
        else decide (10 * 2 ^ 60 ≤ w + p)) := by
    rw [hw, hp, ← trim_c_eq f e hsh, ← trim_half_ulp_eq e hsh]
    rfl
  rw [hu0]
  by_cases heven : f % 2 = 0
  -- Even `f`: round-tripping gives `scale ≤ gap + num`, so the packed sum
  -- cannot fall two units short of the modulus.
  · simp only [heven, reduceIte] at hs
    have hbound : trimScale e ≤ trimGap f e + trimNum e := by
      have hq := (abs_le.mp (hle.mp hs)).2
      exact_mod_cast (by linarith :
        (trimScale e : ℚ) ≤ (trimGap f e : ℚ) + (trimNum e : ℚ))
    split
    · simpa using heven
    · split
      · simpa using heven
      · simp only [decide_eq_true_eq]
        by_contra hcon
        exact absurd (trim_gap_num_lt_scale f e h (by omega)) (by omega)
  -- Odd `f`: round-tripping gives `scale < gap + num`, which excludes both tie
  -- cases and forces the plain comparison.
  · simp only [heven, reduceIte] at hs
    have hbound : trimScale e < trimGap f e + trimNum e := by
      have hq := (abs_lt.mp (hlt.mp hs)).2
      exact_mod_cast (by linarith :
        (trimScale e : ℚ) < (trimGap f e : ℚ) + (trimNum e : ℚ))
    split
    · rename_i htie1
      exact absurd (trim_gap_num_le_scale f e h htie1) (by omega)
    · split
      · rename_i htie0
        exact absurd (trim_gap_num_eq_scale_of_k_zero f e h htie0.1 htie0.2)
          (by omega)
      · simp only [decide_eq_true_eq]
        by_contra hcon
        exact absurd (trim_gap_num_lt_scale f e h (by omega)) (by omega)

-- If the rounding interval contains a multiple of ten, yy trims.
theorem trim_of_coarse_roundtrip (f : ℕ) (e : ℤ) (h : Regular f e) (d : ℕ)
    (h10 : d % 10 = 0)
    (hr : Roundtrips f e (d * 10 ^ decimalExponent e)) :
    ((toDecimalCandidates f e).roundD0
      || (toDecimalCandidates f e).roundU0) = true := by
  rw [Bool.or_eq_true]
  rcases coarse_candidate_cases f e h d h10 hr with rfl | rfl
  · exact Or.inl (round_d0_of_ten_down_roundtrips f e h hr)
  · exact Or.inr (round_u0_of_ten_up_roundtrips f e h hr)

-- Any multiple of ten that round-trips at yy's exponent is what yy emits: it
-- makes yy trim, and trimmed output is a round-tripping multiple of ten too, so
-- uniqueness identifies the two.
theorem yy_eq_of_coarse_roundtrip (f : ℕ) (e : ℤ) (h : Regular f e) (d : ℕ)
    (h10 : d % 10 = 0)
    (hr : Roundtrips f e (d * 10 ^ decimalExponent e)) :
    (toDecimal f e).1 = d := by
  let c := toDecimalCandidates f e
  have htrim : (c.roundD0 || c.roundU0) = true :=
    trim_of_coarse_roundtrip f e h d h10 hr
  -- Trimmed output is a multiple of ten.
  have hten : (toDecimal f e).1 % 10 = 0 := by
    change (if c.roundD0 || c.roundU0 then c.decTen else c.decOne) % 10 = 0
    rw [htrim]
    change (sigHi f e - sigHi f e % 10
      + (if c.roundU0 then 10 else 0)) % 10 = 0
    cases c.roundU0 <;> simp <;> omega
  have hrt : Roundtrips f e
      (((toDecimal f e).1 : ℚ) * 10 ^ decimalExponent e) := yy_roundtrips f e h
  exact coarse_roundtrip_unique f e h _ _ hten h10 hrt hr

-- Full shortness: after removing trailing zeros, yy's result is a shortest
-- decimal representation that round-trips. Any value on the next coarser grid
-- is also a multiple of ten at yy's original exponent. Completeness therefore
-- makes yy select that candidate, and uniqueness forces the reduced significand
-- to have a trailing zero. Reduction rules this out unless the reduced
-- significand is zero, and zero cannot round-trip to a value in the `Regular`
-- range.
theorem yy_shortest (f : ℕ) (e : ℤ) (h : Regular f e) :
    let (d, k) := reduceDecimal (toDecimal f e)
    Shortest f e d k := by
  let y := toDecimal f e
  have hk : y.2 = decimalExponent e := rfl
  have hrt : Roundtrips f e ((y.1 : ℚ) * 10 ^ y.2) := yy_roundtrips f e h
  obtain ⟨t, hkt, hstrip⟩ := reduce_shift y
  have hstop := reduce_reduced y
  have hval := reduce_value y
  rcases hred : reduceDecimal y with ⟨d, k⟩
  simp only [y, hred] at hkt hstrip hstop hval ⊢
  have hone : Roundtrips f e ((d : ℚ) * 10 ^ k) := by rw [hval]; exact hrt
  refine ⟨hone, ?_⟩
  intro d' hr'
  -- On the next coarser grid, `c` is a multiple of ten at yy's exponent.
  let c := d' * 10 ^ (t + 1)
  have hc10 : c % 10 = 0 := by
    simp only [c, pow_succ, ← Nat.mul_assoc]
    exact Nat.mul_mod_left _ _
  have hexp : decimalExponent e + ((t + 1 : ℕ) : ℤ) = k + 1 := by
    rw [hkt, hk]; push_cast; ring
  have hrc : Roundtrips f e ((c : ℚ) * 10 ^ decimalExponent e) := by
    simp only [c]
    rw [ten_pow_shift d' (t + 1), hexp]
    exact hr'
  -- So that candidate is exactly what yy emitted.
  have heq := yy_eq_of_coarse_roundtrip f e h c hc10 hrc
  -- Cancelling the stripped zeros leaves a trailing zero in the reduced
  -- significand.
  have hcancel : d = d' * 10 := by
    apply Nat.mul_right_cancel (m := 10 ^ t) (by positivity)
    rw [← hstrip, heq]
    simp only [c, pow_succ]
    ring
  rcases hstop with hzero | hne
  · rw [hzero] at hone
    exact not_roundtrips_zero f e h (by simpa using hone)
  · rw [hcancel] at hne
    omega
