-- Lean proofs for https://github.com/vitaut/zmij/.
--
-- Copyright (c) 2025 - present, Victor Zverovich
-- Distributed under the MIT license (see LICENSE).

import Core
import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.LinearCombination

/-! # Correctness of Żmij

`Core.lean` proves that an exact, Schubfach-like selection rule produces a
shortest, correctly rounded decimal whenever the decimal grid step is at most
one ULP and strictly greater than a tenth of one. This file proves that Żmij's
regular binary64 path implements that rule. What Żmij shares with any other
algorithm—binary64's spacing, the decimal exponent, the power-of-ten table—is
defined there too.

Żmij scales by `10^(-k-1)` rather than `10^(-k)`, an optimization by Xiang
JunBo: one multiply lands the *shorter* 15-16 digit candidate in the integral
part directly, with no division by ten, and the seventeenth digit is derived
from the fraction left over. So where yy computes a 17-digit significand and
trims a digit off it, Żmij computes the trimmed significand and appends a digit
only when it has to. Like yy and unlike Xiang's xjb64, Żmij uses truncated
power-of-ten significands.

That changes what has to be proved. Żmij's two coarse candidates are the
integral part and its successor, one unit apart on the grid at `k+1`, and its
two rounding flags say which of them lands within half a ULP. The fine
candidate is not a truncation at all but a rounded digit, so the tie rule that
yy wrote into a flag Żmij writes into a rounding constant.

Throughout this file:
* `f`, `e`: binary significand and exponent, denoting `f·2^e`;
* `d`, `k`: decimal significand and exponent, denoting `d·10^k`;
* `h`: the shift `exponentShift e`, aligning `f·2^e` with `10^(-k-1)`.

Żmij makes three decisions, whose correctness is captured by comparisons of
naturals in the integer scale of `## Żmij's arithmetic model`:

    round_down_iff_gap :  roundDown ↔ 2·gap ≤ num
    round_up_iff_gap   :  roundUp   ↔ 2·step ≤ 2·gap + num
    digit_nearest      :  |digitDist| ≤ step

The first two are equivalences: each coarse flag fires exactly when its
candidate lands within half a ULP. The third cannot be: a rounded digit is the
outcome of no comparison. What holds of it is the bound above, that the digit
is nearest on the grid at `k`, with a tie condition making a digit at the bound
even. Both tie rules are written into their bounds: the coarse ones are strict
for odd `f`, resolving a tie in the binary rounding interval, while the digit's
is not, its tie being on the decimal grid and resolved by the digit itself.

The coarse pair carries a second equivalence, on the other side:

    round_down_iff_roundtrips :  roundDown ↔ integralPart round-trips
    round_up_iff_roundtrips   :  roundUp   ↔ integralPart + 1 round-trips

The two boundaries mean different things and stay separate. `*_iff_gap` says the
finite-precision implementation reaches the correct exact decision;
`*_iff_roundtrips` connects the integer scale that decision is made in to the
semantic specification. Consumers read one direction or the other:
`coarse_output_roundtrips` uses `→`, `trim_of_coarse_roundtrip` uses `←`.

Both coarse candidates live on the grid at `k+1`, one unit apart. The reported
significand lives on the grid at `k`, ten times finer, where they become
multiples of ten and the derived digit fills the room between them. Relating the
two grids is all of `## The grid at k`.

## Dependencies

    correct
      ← ulp_scaled_bounds
      ← exact_candidate
          ← coarse_output_roundtrips
          ← fine_output_nearest
          ← trim_of_coarse_roundtrip

    coarse_output_roundtrips, trim_of_coarse_roundtrip
      ← round_down_iff_roundtrips, round_up_iff_roundtrips

    fine_output_nearest
      ← digit_nearest
          ← table_digit_eq

`ulp_scaled_bounds` is the other thing `exact_candidate_correct` asks for: at
Żmij's exponent one ULP spans between one and ten grid steps, so the fine case
has a grid to be correctly rounded on.
-/

namespace Zmij

/-! ## Żmij's conversion

The algorithm, as Żmij computes it. One 192-bit multiply of the significand by
the 128-bit power of ten, keeping the top 128 bits; the integral part and the
fraction are read out of that product at a fixed bit position, and the three
decisions are comparisons on those two values.

The extra shift is the nine bits of headroom Żmij leaves below the integral
part, which `exponentShift` folds in. Nine is not forced: 3 keeps the shift
non-negative, 10 keeps `f·2^h` inside 64 bits, and 9 lets the digit constant be
shared with the base-ten multiply. What the proof needs from it is only that the
shift stays in `[6, 9]`.
-/

/-- Shift chosen to align the binary exponent with the power of ten, including
    Żmij's nine bits of headroom. -/
def exponentShift (e : ℤ) : ℕ :=
  Int.toNat (e + (-(binary64.decimalExponent e + 1) * 217_707) / 2 ^ 16 + 10)

/-- The top 128 bits of Żmij's 192-bit product: `⌊f·2^h·p10 / 2^64⌋`. -/
def scaledSignificand (f : ℕ) (e : ℤ) : ℕ :=
  f * 2 ^ exponentShift e
    * binary64.power10Significand (-(binary64.decimalExponent e + 1)) / 2 ^ 64

/-- The integral part of the scaled value, Żmij's shorter candidate: the product
    with the nine headroom bits and the fraction shifted off. -/
def integralPart (f : ℕ) (e : ℤ) : ℕ := scaledSignificand f e / 2 ^ 73

/-- The fraction of the scaled value, in units of `2^-64`. -/
def fractionalPart (f : ℕ) (e : ℤ) : ℕ :=
  scaledSignificand f e / 2 ^ 9 % 2 ^ 64

/-- Half a ULP in the same units, truncated to the top 64 bits of the power of
    ten, plus one when `f` is even. That `+1` is the tie rule: it turns each of
    the two comparisons below from strict into non-strict exactly when a tie is
    allowed to round. -/
def halfUlp (f : ℕ) (e : ℤ) : ℕ :=
  binary64.power10Significand (-(binary64.decimalExponent e + 1)) / 2 ^ 64
      / 2 ^ (10 - exponentShift e)
    + (1 - f % 2)

/-- Rounding constant for the derived digit: half of `2^64`, nudged up by six.
    The nudge covers the truncation in `fractionalPart`; it moves the digit
    boundary by less than one of its units. -/
def biasedHalf : ℕ := 2 ^ 63 + 6

/-- The derived digit as a rounding rule: ten times the fraction, rounded with
    `biasedHalf`. The one exact half that survives truncation is a quarter,
    whose digit 2.5 has to go to even. Żmij computes the digit from
    `digitOffset` instead, and `table_digit_eq` shows the two agree. -/
def roundedDigit (frac : ℕ) : ℕ :=
  if frac = 2 ^ 62 then 2 else (frac * 10 + biasedHalf) / 2 ^ 64

/-- Żmij's `digit_offsets`: the digit is the top four bits of the fraction plus
    the offset of its cell, the top four bits of the fraction itself. Each cell
    holds at most one digit boundary, and its offset carries into the top four
    bits exactly from that boundary on. -/
def digitOffset : ℕ → ℕ
  | 0 => 0x0333333333333333
  | 1 => 0
  | 2 => 0xf999999999999999
  | 3 => 0xf000000000000000
  | 4 => 0xefffffffffffffff
  | 5 => 0xe666666666666667
  | 6 => 0xe000000000000000
  | 7 => 0xdccccccccccccccd
  | 8 => 0xd333333333333333
  | 9 => 0xd000000000000000
  | 10 => 0xc99999999999999a
  | 11 => 0xc000000000000000
  | 12 => 0xc000000000000000
  | 13 => 0xb666666666666666
  | 14 => 0xb000000000000000
  | 15 => 0xaccccccccccccccc
  | _ => 0

structure DecimalCandidates where
  k : ℤ
  integral : ℕ
  digit : ℕ
  roundUp : Bool
  roundDown : Bool

def toDecimalCandidates (f : ℕ) (e : ℤ) : DecimalCandidates :=
  let frac := fractionalPart f e
  let half := halfUlp f e

  -- Żmij detects both of these by 64-bit carry: `frac + half` wrapping past
  -- `2^64` is the fraction reaching up to the next integer.
  let roundUp : Bool := 2 ^ 64 ≤ frac + half
  let roundDown : Bool := frac < half

  -- The digit from the fraction's cell, its top four bits, which Żmij reads
  -- from the high word of the product alone: the four bits below the integral
  -- part.
  let cell := scaledSignificand f e / 2 ^ 69 % 16
  let digit : ℕ := (frac + digitOffset cell) % 2 ^ 64 / 2 ^ 60

  { k := binary64.decimalExponent e
    integral := integralPart f e + (if roundUp then 1 else 0)
    digit := digit
    roundUp := roundUp
    roundDown := roundDown }

/-- Converts a regularly spaced binary64 value f·2^e to a decimal significand
    and exponent using Żmij's regular path. The significand is reported on the
    grid at `k`, so the shorter candidate appears as a multiple of ten. -/
def toDecimal (f : ℕ) (e : ℤ) : ℕ × ℤ :=
  let c := toDecimalCandidates f e
  (c.integral * 10 + (if c.roundUp || c.roundDown then 0 else c.digit), c.k)

/-! ### The shift

The shift is `e + pe + 9`, where `pe` is the exponent of `10^(-k-1)`, and it
lands in `[6, 9]`. Both facts are about the two fixed-point constants rather
than about magnitudes, so both are checked over the exponent range.
-/

/-- The shift before clamping, which is what the checks below enumerate. -/
theorem exponent_shift_eq (e : ℤ) :
    (exponentShift e : ℤ) = max 0
      (e + (-(binary64.decimalExponent e + 1) * 217_707) / 2 ^ 16 + 10) := by
  rw [exponentShift]
  omega

/-- The shift is at least six and at most nine. The upper bound is what keeps
    `f·2^h` inside 64 bits in the implementation; the lower bound is what makes
    `10 - h` a positive shift in `halfUlp`. -/
theorem exponent_shift_bounds :
    ∀ e ∈ Finset.Icc (-1074 : ℤ) 971,
      6 ≤ e + (-(binary64.decimalExponent e + 1) * 217_707) / 2 ^ 16 + 10 ∧
        e + (-(binary64.decimalExponent e + 1) * 217_707) / 2 ^ 16 + 10
          ≤ 9 := by
  -- This and the two checks like it below enumerate 2046 exponents each.
  -- `+kernel` keeps them out of the elaborator, whose recursion and
  -- exponentiation guards they would otherwise trip.
  decide +kernel

/-- Hence the shift, as a natural, is in `[6, 9]`. -/
theorem exponent_shift_range (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    6 ≤ exponentShift e ∧ exponentShift e ≤ 9 := by
  have h := exponent_shift_bounds e (by simpa [Finset.mem_Icc] using he)
  have := exponent_shift_eq e
  omega

/-- The shift undoes the power-of-ten exponent: `h - 9 - pe = e`. Both sides
    scale the same fixed-point quotient, so once the shift is known not to have
    been clamped this is arithmetic, whatever the decimal exponent is. -/
theorem exponent_shift_align (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    (exponentShift e : ℤ) - 9
      - binary64.power10Exponent (-(binary64.decimalExponent e + 1)) = e := by
  have hb := exponent_shift_bounds e (by simpa [Finset.mem_Icc] using he)
  have heq := exponent_shift_eq e
  simp only [Format.power10Exponent, binary64] at hb heq ⊢
  omega

/-! ## Żmij's arithmetic model

Żmij's three decisions are comparisons of 64-bit integers. What follows
rebuilds them as exact naturals: the scale they are read in, what their
truncations discard, and how that scale relates to the rationals the
specification is stated over.
-/

/-! ### The integer quantities

The power of ten is a ratio `num / den`, and clearing that denominator turns
every comparison below into one between naturals. In integers, one step of the
grid at `k+1` is `2^(137-h)·den`, one ULP of the value is exactly `num`, and the
fraction Żmij compares is measured in units of `unit·den = 2^(73-h)·den`.

Two divisions of the implementation compose into one here. `integralPart`
shifts the 192-bit product down by 64 and then by 73, which is a single
division by `2^(137-h)` once the shift is factored out, and `fractionalPart`
reads the same quotient 64 bits lower.
-/

/-- Numerator of the exact power of ten Żmij multiplies by. -/
def num (e : ℤ) : ℕ :=
  binary64.power10Num (-(binary64.decimalExponent e + 1))

/-- Its denominator. -/
def den (e : ℤ) : ℕ :=
  binary64.power10Den (-(binary64.decimalExponent e + 1))

/-- Its 128-bit truncation, the `p10` of the implementation. -/
def p10 (e : ℤ) : ℕ :=
  binary64.power10Significand (-(binary64.decimalExponent e + 1))

theorem den_pos (e : ℤ) : 0 < den e := binary64.power10_den_pos _

/-- The truncation is natural-number division, which is what keeps this whole
    layer in `Nat`. -/
theorem p10_nat (e : ℤ) : p10 e = num e / den e :=
  binary64.power10_significand_nat _

/-- `den·p10 + τ = num`: the truncated power of ten and the bits it dropped. -/
theorem num_split (e : ℤ) : den e * p10 e + num e % den e = num e := by
  rw [p10_nat]; exact Nat.div_add_mod _ _

/-- The unit the fraction is measured in, before the denominator enters. -/
def unit (e : ℤ) : ℕ := 2 ^ (73 - exponentShift e)

theorem unit_pos (e : ℤ) : 0 < unit e := by rw [unit]; positivity

/-- One step of the coarse grid at `k+1`, in the integer scale. -/
def step (e : ℤ) : ℕ := unit e * 2 ^ 64 * den e

theorem step_pos (e : ℤ) : 0 < step e := by
  rw [step]
  exact Nat.mul_pos (Nat.mul_pos (unit_pos e) (by positivity)) (den_pos e)

/-- Both readouts are one division of the integer product, by whatever the
    implementation's two shifts compose into: the alignment shift divides out of
    both sides of it. -/
private theorem scaled_div (f : ℕ) (e : ℤ) {n m : ℕ}
    (h : (2 : ℕ) ^ 64 * 2 ^ n = 2 ^ exponentShift e * m) :
    scaledSignificand f e / 2 ^ n = f * p10 e / m := by
  rw [scaledSignificand, ← p10, Nat.div_div_eq_div_mul, h,
    show f * 2 ^ exponentShift e * p10 e
        = 2 ^ exponentShift e * (f * p10 e) from by ring,
    Nat.mul_div_mul_left _ _ (by positivity)]

/-- The integral part is the quotient by one coarse step. -/
theorem integral_quotient (f : ℕ) (e : ℤ) (hs : exponentShift e ≤ 9) :
    integralPart f e = f * p10 e / (unit e * 2 ^ 64) := by
  rw [integralPart]
  refine scaled_div f e ?_
  rw [unit, ← pow_add, ← pow_add, ← pow_add]
  congr 1
  omega

/-- The fraction is the same quotient read 64 bits lower. -/
theorem fraction_quotient (f : ℕ) (e : ℤ) (hs : exponentShift e ≤ 9) :
    fractionalPart f e = f * p10 e / unit e % 2 ^ 64 := by
  have h : scaledSignificand f e / 2 ^ 9 = f * p10 e / unit e := by
    refine scaled_div f e ?_
    rw [unit, ← pow_add, ← pow_add]
    congr 1
    omega
  rw [fractionalPart, h]

/-- What that division left behind, the residue of the integer product in one
    coarse step. The fraction is its top and the truncation its bottom, and
    both of the digit's boundaries constrain the two together, so both are
    stated about this. -/
def res (f : ℕ) (e : ℤ) : ℕ := f * p10 e % (unit e * 2 ^ 64)

/-- The gap from the integral part up to the exact scaled value, in integers:
    what the quotient dropped, plus what the truncated power of ten dropped. -/
def gap (f : ℕ) (e : ℤ) : ℕ := den e * res f e + f * (num e % den e)

/-- The integral part scaled back up, plus the gap, is the scaled value
    `f·num`: `Nat.div_add_mod` recovers the product from the quotient and
    `num_split` the power of ten from its truncation. -/
theorem integral_add_gap (f : ℕ) (e : ℤ) (hs : exponentShift e ≤ 9) :
    integralPart f e * step e + gap f e = f * num e := by
  rw [integral_quotient f e hs, step, gap, res]
  calc f * p10 e / (unit e * 2 ^ 64) * (unit e * 2 ^ 64 * den e)
        + (den e * (f * p10 e % (unit e * 2 ^ 64))
          + f * (num e % den e))
      = den e * (f * p10 e / (unit e * 2 ^ 64) * (unit e * 2 ^ 64)
          + f * p10 e % (unit e * 2 ^ 64)) + f * (num e % den e) := by ring
    _ = den e * (f * p10 e) + f * (num e % den e) := by
          rw [Nat.div_add_mod']
    _ = f * (den e * p10 e + num e % den e) := by ring
    _ = f * num e := by rw [num_split]

/-! ### What the truncations discard

Both of Żmij's comparisons are between a truncated 64-bit value and the exact
quantity it stands for, so each needs the size of what was dropped. The gap is
the fraction in integer units plus a remainder below one such unit, and one
ULP is twice half a ULP in the same units plus a remainder below one. Those two
remainders are the whole error budget the certificates below have to close.
-/

/-- The residue splits at the fraction's own unit: the fraction above it,
    the remainder the fraction dropped below. -/
theorem res_split (f : ℕ) (e : ℤ) (hs : exponentShift e ≤ 9) :
    res f e = unit e * fractionalPart f e + f * p10 e % unit e := by
  have hdiv : f * p10 e % (unit e * 2 ^ 64) / unit e
      = f * p10 e / unit e % 2 ^ 64 := Nat.mod_mul_right_div_self _ _ _
  have hmod : f * p10 e % (unit e * 2 ^ 64) % unit e = f * p10 e % unit e :=
    Nat.mod_mod_of_dvd _ (dvd_mul_right _ _)
  have hsplit := Nat.div_add_mod (f * p10 e % (unit e * 2 ^ 64)) (unit e)
  rw [hdiv, hmod] at hsplit
  rw [res, ← hsplit, fraction_quotient f e hs]

/-- So the gap, in the units the fraction is measured in, is the fraction plus
    what the two truncations discarded. -/
theorem gap_eq_fraction_add (f : ℕ) (e : ℤ) (hs : exponentShift e ≤ 9) :
    gap f e
      = unit e * den e * fractionalPart f e
        + (den e * (f * p10 e % unit e) + f * (num e % den e)) := by
  rw [gap, res_split f e hs]
  ring

/-! ### Crossing into ℚ

The specification is about rationals; everything above is about naturals.
`step` is exactly the factor that clears the denominator, and it sends one
scaled ULP to `num`, so a candidate's distance from the value is always an
integer and every comparison the specification asks for is a comparison of
integers. This is the only place `ℚ` appears.
-/

/-- One coarse step is the alignment shift with the denominator: the fraction's
    unit and the 64 bits above it compose into `2^(137-h)`. -/
theorem step_pow (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    step e = 2 ^ (137 - exponentShift e) * den e := by
  rw [step, unit, ← pow_add,
    show 73 - exponentShift e + 64 = 137 - exponentShift e from by
      have := exponent_shift_range e he; omega]

/-- `step` clears the denominator in `power10_exact_ratio`, leaving `num`
    times the binary-decimal scaling factor. -/
theorem step_eq (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    (step e : ℚ)
      = (num e : ℚ) * (2 ^ (-e) * 10 ^ (binary64.decimalExponent e + 1)) := by
  set k := binary64.decimalExponent e
  set pe := binary64.power10Exponent (-(k + 1))
  have hd : (0 : ℚ) < (den e : ℚ) := by exact_mod_cast den_pos e
  have hnum : (10 : ℚ) ^ (-(k + 1)) * 2 ^ (128 - pe) * den e = num e := by
    rw [show (128 : ℤ) = (binary64.p10Width : ℤ) from rfl,
      binary64.power10_exact_ratio, ← num, ← den,
      div_mul_cancel₀ _ (ne_of_gt hd)]
  -- The inverse scale turns the power-of-ten factor into `2^(137-h)`, which is
  -- where the shift alignment is spent.
  have hscale : (10 : ℚ) ^ (-(k + 1)) * 2 ^ (128 - pe)
        * (2 ^ (-e) * 10 ^ (k + 1))
      = 2 ^ (137 - exponentShift e) := by
    have h10 : (10 : ℚ) ^ (-(k + 1)) * 10 ^ (k + 1) = 1 := by
      rw [← zpow_add₀ (by norm_num : (10 : ℚ) ≠ 0)]; simp
    have halign : (exponentShift e : ℤ) - 9 - pe = e := exponent_shift_align e he
    have hs := exponent_shift_range e he
    calc (10 : ℚ) ^ (-(k + 1)) * 2 ^ (128 - pe) * (2 ^ (-e) * 10 ^ (k + 1))
        = (10 ^ (-(k + 1)) * 10 ^ (k + 1)) * (2 ^ (128 - pe) * 2 ^ (-e)) := by
          ring
      _ = (2 : ℚ) ^ ((128 - pe) + -e) := by
          rw [h10, one_mul, ← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
      _ = 2 ^ (137 - exponentShift e) := by
          rw [show (128 - pe) + -e = ((137 - exponentShift e : ℕ) : ℤ)
                from by omega, zpow_natCast]
  rw [step_pow e he]
  push_cast
  rw [← hscale, ← hnum]
  ring

/-- The scale sends half a scaled ULP to half of `num`, so twice the scale
    sends it to `num` itself: one ULP on the grid at `k+1` is exactly one
    `num`. -/
theorem half_ulp_scaled (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    ulp e * 10 ^ (-(binary64.decimalExponent e + 1)) / 2
        * (((2 : ℕ) : ℚ) * (step e : ℚ))
      = ((num e : ℕ) : ℚ) := by
  set k := binary64.decimalExponent e
  have h10 : (10 : ℚ) ^ (-(k + 1)) * 10 ^ (k + 1) = 1 := by
    rw [← zpow_add₀ (by norm_num : (10 : ℚ) ≠ 0)]; simp
  have h2 : (2 : ℚ) ^ e * 2 ^ (-e) = 1 := by
    rw [← zpow_add₀ (two_ne_zero' ℚ)]; simp
  calc ulp e * 10 ^ (-(k + 1)) / 2 * (((2 : ℕ) : ℚ) * (step e : ℚ))
      = (num e : ℚ) * (2 ^ e * 2 ^ (-e))
          * (10 ^ (-(k + 1)) * 10 ^ (k + 1)) := by
        rw [ulp, step_eq e he]; push_cast; ring
    _ = ((num e : ℕ) : ℚ) := by rw [h10, h2]; ring

/-- The one Żmij-specific fact the generic bridge needs: `step` sends the
    scaled value to the integer `f·num`, so every candidate distance is an
    integer. -/
theorem value_scaled (f : ℕ) (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    value f e * 10 ^ (-(binary64.decimalExponent e + 1)) * (step e : ℚ)
      = ((f * num e : ℤ) : ℚ) := by
  have h := half_ulp_scaled e he
  rw [ulp] at h
  push_cast at h ⊢
  rw [value]
  linear_combination (f : ℚ) * h

/-- A candidate on the grid at `k+1` round-trips exactly when twice its signed
    distance stays within one ULP, that is within `num`, strictly so for odd
    `f`. This is the only use either direction of the coarse argument makes of
    `ℚ`. -/
theorem roundtrips_iff_dist (f : ℕ) (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) {c : ℕ}
    {dist : ℤ} (hc : (c : ℤ) * step e + dist = f * num e) :
    Roundtrips f e (c * 10 ^ (binary64.decimalExponent e + 1))
      ↔ if f % 2 = 0 then -(num e : ℤ) ≤ 2 * dist ∧ 2 * dist ≤ num e
        else -(num e : ℤ) < 2 * dist ∧ 2 * dist < num e := by
  obtain ⟨hle, hlt, -⟩ := scaled_cmp_of_int_eq (c := c) (scale := step e)
    (a := 2) (b := num e) (dist := dist)
    (x := value f e * 10 ^ (-(binary64.decimalExponent e + 1)))
    (thr := ulp e * 10 ^ (-(binary64.decimalExponent e + 1)) / 2)
    (step_pos e) two_pos (value_scaled f e he) (half_ulp_scaled e he) hc
  refine (roundtrips_iff_scaled f e (binary64.decimalExponent e + 1) c).trans ?_
  split_ifs
  · exact hle.trans (by omega)
  · exact hlt.trans (by omega)

/-! ## The error budget

Both coarse flags compare Żmij's truncated fraction with its truncated half
ULP. In integers, one fraction unit is `edge = 2^(73-h)·den`: the fraction
stands for that much of the gap apiece, the half ULP for twice that much of one
ULP apiece, and each of the two truncations leaves a remainder below one unit,
`err` from the gap and `errHalf` from the ULP. Those two remainders are the
whole budget the certificates below have to close.
-/

/-- One fraction unit in the integer scale: the resolution both coarse
    comparisons work at. -/
def edge (e : ℤ) : ℕ := unit e * den e

theorem edge_pos (e : ℤ) : 0 < edge e :=
  Nat.mul_pos (unit_pos e) (den_pos e)

/-- One coarse step is `2^64` fraction units, which is what makes the carry out
    of Żmij's 64-bit addition the fraction reaching the next integer. -/
theorem step_eq_edge (e : ℤ) : step e = edge e * 2 ^ 64 := by
  rw [step, edge]
  ring

/-- Half a ULP as Żmij truncates it, in fraction units. -/
def half (e : ℤ) : ℕ := p10 e / (2 * unit e)

/-- What the fraction discards from the gap: the bits below one fraction unit,
    plus the power-of-ten truncation the whole product carries. -/
def err (f : ℕ) (e : ℤ) : ℕ :=
  den e * (f * p10 e % unit e) + f * (num e % den e)

/-- What the truncated half ULP discards from one ULP. -/
def errHalf (e : ℤ) : ℕ :=
  den e * (p10 e % (2 * unit e)) + num e % den e

/-- Żmij's `halfUlp` is the truncated half ULP plus one for even `f`. That
    `+1` is the tie rule: it turns each comparison below from strict into
    non-strict exactly when a tie is allowed to round. -/
theorem half_ulp_eq (f : ℕ) (e : ℤ) (hs : exponentShift e ≤ 9) :
    halfUlp f e = half e + (1 - f % 2) := by
  have hpow : (2 : ℕ) ^ 64 * 2 ^ (10 - exponentShift e) = 2 * unit e := by
    rw [unit, ← pow_add,
      show 64 + (10 - exponentShift e) = 73 - exponentShift e + 1 from by omega,
      pow_succ]
    ring
  rw [halfUlp, half, p10, Nat.div_div_eq_div_mul, hpow]

/-- The gap is the fraction in integer units plus what the fraction
    discarded. -/
theorem gap_eq_edge_fraction (f : ℕ) (e : ℤ) (hs : exponentShift e ≤ 9) :
    gap f e = edge e * fractionalPart f e + err f e := by
  rw [edge, err]
  exact gap_eq_fraction_add f e hs

/-- One ULP is twice the truncated half ULP in integer units, plus what that
    truncation discarded. -/
theorem num_eq_edge_half (e : ℤ) :
    num e = 2 * (edge e * half e) + errHalf e := by
  rw [edge, half, errHalf]
  calc num e = den e * p10 e + num e % den e := (num_split e).symm
    _ = den e * (2 * unit e * (p10 e / (2 * unit e))
          + p10 e % (2 * unit e)) + num e % den e := by
        rw [Nat.div_add_mod]
    _ = 2 * (unit e * den e * (p10 e / (2 * unit e)))
          + (den e * (p10 e % (2 * unit e)) + num e % den e) := by ring

/-- The half-ULP truncation discards less than two fraction units, so the
    fraction reaching one unit past the half ULP settles the comparison by
    itself. -/
theorem err_half_lt (e : ℤ) : errHalf e < 2 * edge e := by
  have hden := den_pos e
  have h1 : p10 e % (2 * unit e) + 1 ≤ 2 * unit e :=
    Nat.mod_lt _ (by have := unit_pos e; omega)
  have h2 : num e % den e < den e := Nat.mod_lt _ hden
  have hmono : den e * (p10 e % (2 * unit e) + 1) ≤ den e * (2 * unit e) :=
    Nat.mul_le_mul_left _ h1
  have hexp : den e * (p10 e % (2 * unit e) + 1)
      = den e * (p10 e % (2 * unit e)) + den e := by ring
  have hedge : 2 * edge e = den e * (2 * unit e) := by rw [edge]; ring
  rw [errHalf, hedge]
  omega

/-- Everything about the truncated power of ten that has to be checked per
    exponent rather than derived from magnitudes: the fraction's error stays
    within two fraction units of the half ULP's, and the two together stay
    under four. Both are stated at the largest error any significand can carry,
    and the first is tight, with two to spare at `e = -90`. -/
def checksHold (e : ℤ) : Bool :=
  let d := den e
  let u := unit e
  let tau := num e % d
  let eHalf := d * (num e / d % (2 * u)) + tau
  let eMax := d * (u - 1) + (2 ^ 53 - 1) * tau
  decide (2 * eMax < eHalf + 2 * (u * d)
    ∧ 2 * eMax + eHalf < 4 * (u * d))

theorem checks_all :
    ∀ e ∈ Finset.Icc (-1074 : ℤ) 971, checksHold e = true := by
  decide +kernel

/-- Those two checks for one exponent, with the truncation read back as
    `p10`. -/
theorem checks (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    2 * (den e * (unit e - 1) + (2 ^ 53 - 1) * (num e % den e))
        < errHalf e + 2 * edge e
      ∧ 2 * (den e * (unit e - 1) + (2 ^ 53 - 1) * (num e % den e))
          + errHalf e < 4 * edge e := by
  have hcert := checks_all e (by simpa [Finset.mem_Icc] using he)
  simp only [checksHold, decide_eq_true_eq] at hcert
  rw [errHalf, edge, p10_nat]
  exact hcert

/-- The fraction's truncation error, at its largest over the significands. -/
theorem err_le (f : ℕ) (e : ℤ) (hr : binary64.Regular f e) :
    err f e ≤ den e * (unit e - 1) + (2 ^ 53 - 1) * (num e % den e) := by
  have h1 : f * p10 e % unit e ≤ unit e - 1 := by
    have := Nat.mod_lt (f * p10 e) (unit_pos e)
    omega
  have h2 : f ≤ 2 ^ 53 - 1 := by have : f < 2 ^ 53 := hr.sig_lt; omega
  rw [err]
  exact Nat.add_le_add (Nat.mul_le_mul_left _ h1) (Nat.mul_le_mul_right _ h2)

/-- The two error bounds the comparisons are decided by, at this
    significand. -/
theorem err_bounds (f : ℕ) (e : ℤ) (hr : binary64.Regular f e) :
    2 * err f e < errHalf e + 2 * edge e
      ∧ 2 * err f e + errHalf e < 4 * edge e := by
  have hle := err_le f e hr
  have hchk := checks e hr.range
  omega

/-- The power of ten is normalized in integer form too. -/
theorem num_bounds (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    2 ^ 127 * den e ≤ num e ∧ num e < 2 ^ 128 * den e :=
  binary64.power10_ratio_normalized he.1 he.2
    (-(binary64.decimalExponent e + 1)) (by omega)

/-- In particular one ULP is positive, which is what makes both coarse
    boundaries genuine ones. -/
theorem num_pos (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) : 0 < num e := by
  have hlow := (num_bounds e he).1
  have h : 0 < 2 ^ 127 * den e := by
    have := den_pos e
    positivity
  omega

/-- A fraction unit is negligible against the power of ten, which is what
    leaves both coarse boundaries well inside the two steps the residue below
    runs over: `edge ≤ 2^67·den` while `num ≥ 2^127·den`. -/
theorem edge_bounds (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    4 * edge e < num e ∧ num e + 4 * edge e < 2 * step e := by
  obtain ⟨hs6, hs9⟩ := exponent_shift_range e he
  obtain ⟨hlo, hhi⟩ := num_bounds e he
  have hden := den_pos e
  have hedge : 4 * edge e ≤ 2 ^ 69 * den e := by
    rw [edge]
    calc 4 * (unit e * den e) ≤ 4 * (2 ^ 67 * den e) :=
          Nat.mul_le_mul_left _ (Nat.mul_le_mul_right _
            (by rw [unit]; exact Nat.pow_le_pow_right (by norm_num) (by omega)))
      _ = 2 ^ 69 * den e := by ring
  have hmul : 2 ^ 128 * den e ≤ step e := by
    rw [step]
    calc 2 ^ 128 * den e = 2 ^ 64 * 2 ^ 64 * den e := by ring
      _ ≤ unit e * 2 ^ 64 * den e :=
          Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _
            (by rw [unit]; exact Nat.pow_le_pow_right (by norm_num) (by omega)))
  have h69 : 2 ^ 69 * den e < 2 ^ 127 * den e :=
    mul_lt_mul_of_pos_right (by norm_num) hden
  omega

/-! ### Refuting the exceptional windows

Everything above is analytic, and it decides both comparisons except within one
fraction unit of a boundary. What is left is a band of that width either side of
each boundary, minus the boundary itself; the certificates say those bands are
empty, which is what leaves the exact ties their room.

Both boundaries share one modular problem. In integers, one coarse step is
`step` and the doubled gap is the residue of `2·num·f` in two steps, so the
trim-down boundary is `num` and the trim-up boundary `2·step - num`. A violation
of either puts that residue in a window of relative width about `2^-63`, which
is the kind of question `ModWindows` answers with one multiplier per exponent.

Ties are not rare here and not refutable: `10^(-k-1)` for `k` in `[0, 22]` gives
a scaled ULP with a small denominator, and the boundary is then hit by a whole
residue class of significands, up to a fifth of them at `k = 0`. What makes
those cases correct is that each is an exact tie, resolved by parity, which is
exactly what excluding the boundary from the windows says.
-/

/-- The gap is under one coarse step by all but the errors' reach: the fraction
    accounts for all but one unit of the step, and the two truncations for less
    than one unit plus `2^53` denominators. -/
theorem gap_lt_step_add (f : ℕ) (e : ℤ) (hr : binary64.Regular f e) :
    gap f e < step e + 2 ^ 53 * den e := by
  have hs := (exponent_shift_range e hr.range).2
  have hden := den_pos e
  have hτ : num e % den e < den e := Nat.mod_lt _ hden
  have hgap := gap_eq_edge_fraction f e hs
  -- The fraction leaves one unit of the step unaccounted for.
  have hfrac : fractionalPart f e < 2 ^ 64 := by
    rw [fractionalPart]; exact Nat.mod_lt _ (by positivity)
  have hfr : edge e * fractionalPart f e + edge e ≤ step e := by
    rw [step_eq_edge]
    calc edge e * fractionalPart f e + edge e
        = edge e * (fractionalPart f e + 1) := by ring
      _ ≤ edge e * 2 ^ 64 := Nat.mul_le_mul_left _ (by omega)
  -- Both truncations together stay under one unit plus the error's reach.
  have h1 : den e * (f * p10 e % unit e) < edge e := by
    rw [edge, Nat.mul_comm (unit e) (den e)]
    exact mul_lt_mul_of_pos_left (Nat.mod_lt _ (unit_pos e)) hden
  have h2 : f * (num e % den e) < 2 ^ 53 * den e :=
    Nat.mul_lt_mul'' (by have : f < 2 ^ 53 := hr.sig_lt; omega) hτ
  rw [err] at hgap
  omega

/-- The doubled gap read modulo two coarse steps. The truncated power of ten
    can leave the integral part one short of the exact one, and then the gap is
    a whole step and this residue wraps to zero. -/
def rest (f : ℕ) (e : ℤ) : ℕ :=
  if step e ≤ gap f e then gap f e - step e else gap f e

private theorem mod_of_add_mul {a q r m : ℕ} (h : a = m * q + r) (hlt : r < m) :
    a % m = r := by
  rw [h, Nat.mul_add_mod, Nat.mod_eq_of_lt hlt]

/-- The doubled gap, less a whole step where it took one, is the residue of
    `2·num·f` in two steps. -/
theorem rest_mod (f : ℕ) (e : ℤ) (hr : binary64.Regular f e) :
    2 * num e * f % (2 * step e) = 2 * rest f e := by
  have hid := integral_add_gap f e (exponent_shift_range e hr.range).2
  have hden := den_pos e
  have hpow : 2 ^ 53 * den e ≤ step e := by
    rw [step]
    exact Nat.mul_le_mul_right _
      (calc (2 : ℕ) ^ 53 ≤ 2 ^ 64 := Nat.pow_le_pow_right (by norm_num) (by omega)
        _ ≤ unit e * 2 ^ 64 := Nat.le_mul_of_pos_left _ (unit_pos e))
  have hwide := gap_lt_step_add f e hr
  rw [rest]
  split_ifs with h
  · obtain ⟨d, hd⟩ : ∃ d, gap f e = step e + d := ⟨gap f e - step e, by omega⟩
    rw [show gap f e - step e = d from by omega]
    refine mod_of_add_mul (q := integralPart f e + 1) ?_ (by omega)
    calc 2 * num e * f = 2 * (integralPart f e * step e + gap f e) := by
          rw [hid]; ring
      _ = 2 * step e * (integralPart f e + 1) + 2 * d := by rw [hd]; ring
  · refine mod_of_add_mul (q := integralPart f e) ?_ (by omega)
    calc 2 * num e * f = 2 * (integralPart f e * step e + gap f e) := by
          rw [hid]; ring
      _ = 2 * step e * integralPart f e + 2 * gap f e := by ring

/-- The residues the error bounds cannot decide: one fraction unit's reach
    either side of each coarse boundary, the boundaries themselves excluded,
    plus the overshoot of a whole step, which no significand reaches either. -/
private def expWindows (e : ℤ) : ModWindows :=
  binary64.regularWindows (2 * num e) (2 * step e) e <|
    let n : ℤ := num e
    let m : ℤ := 2 * step e
    let w : ℤ := 4 * edge e
    [(1, w), (n - w, n - 1), (n + 1, n + w),
      (m - n - w, m - n - 1), (m - n + 1, m - n + w)]

/-- Close `∃ q, (expWindows e).refutedBy q = true` for a literal exponent. -/
elab "exp_cert" : tactic => modCertTactic fun e => (expWindows e).search

private theorem exp_windows_refuted (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    ∃ q, (expWindows e).refutedBy q = true := by
  obtain ⟨hlo, hhi⟩ := he
  interval_cases e <;> exp_cert

/-- A doubled gap landing in a refuted window is impossible: it is the residue
    of `2·num·f` modulo two coarse steps. -/
private theorem no_window_hit {rmin rmax q : ℤ} (f : ℕ) (e : ℤ)
    (hr : binary64.Regular f e)
    (hcert : (expWindows e).refutedBy q = true)
    (hmem : (rmin, rmax) ∈ (expWindows e).windows)
    (hrmin : rmin ≤ 2 * (rest f e : ℤ)) (hrmax : 2 * (rest f e : ℤ) ≤ rmax) :
    False :=
  binary64.regular_not_hit f hr (by have := step_pos e; omega) hcert hmem
    (rest_mod f e hr).symm (by push_cast; omega) (by push_cast; omega)

/-- The gap never passes a whole coarse step. The truncated power of ten can put
    the integral part one short of the exact one, which happens exactly when the
    scaled value is an integer, and Żmij's carry recovers it; anything beyond
    that would be an overshoot in the first window. -/
theorem gap_le_step (f : ℕ) (e : ℤ) (hr : binary64.Regular f e) :
    gap f e ≤ step e := by
  by_contra hcon
  obtain ⟨q, hcert⟩ := exp_windows_refuted e hr.range
  have hs := (exponent_shift_range e hr.range).2
  have hwide := gap_lt_step_add f e hr
  have hrest : rest f e = gap f e - step e := by
    rw [rest]
    split_ifs with h
    · rfl
    · omega
  -- The overshoot is under `2^53` denominators, well inside one fraction unit's
  -- reach, which is what the first window covers.
  have hreach : 2 ^ 54 * den e ≤ 4 * edge e := by
    rw [edge]
    calc 2 ^ 54 * den e ≤ 2 ^ 64 * den e :=
          Nat.mul_le_mul_right _ (Nat.pow_le_pow_right (by norm_num) (by omega))
      _ ≤ 4 * (unit e * den e) := by
          rw [unit, show (4 : ℕ) = 2 ^ 2 from by norm_num, ← Nat.mul_assoc,
            ← pow_add]
          exact Nat.mul_le_mul_right _
            (Nat.pow_le_pow_right (by norm_num) (by omega))
  exact no_window_hit f e hr hcert (.head _) (by rw [hrest]; omega)
    (by rw [hrest]; omega)

/-- With the gap inside a step, the residue is the doubled gap itself, except
    where the gap is exactly a step and it wraps to zero. -/
theorem rest_eq (f : ℕ) (e : ℤ) (hr : binary64.Regular f e) :
    rest f e = gap f e ∨ (gap f e = step e ∧ rest f e = 0) := by
  have hle := gap_le_step f e hr
  rw [rest]
  split_ifs with h
  · exact Or.inr ⟨by omega, by omega⟩
  · exact Or.inl rfl

/-- Either the doubled gap sits exactly on the boundary `b`, a genuine exact
    tie, or it is more than one fraction unit's reach away from it. A gap of
    exactly one step puts the doubled gap at `2·step`, past every window, so
    interiority covers that case rather than a certificate. -/
private theorem gap_tie_or_far {b : ℤ} (f : ℕ) (e : ℤ)
    (hr : binary64.Regular f e)
    (hb : 4 * (edge e : ℤ) < b ∧ b + 4 * (edge e : ℤ) < 2 * (step e : ℤ))
    (hbelow : (b - 4 * (edge e : ℤ), b - 1) ∈ (expWindows e).windows)
    (habove : (b + 1, b + 4 * (edge e : ℤ)) ∈ (expWindows e).windows) :
    2 * (gap f e : ℤ) = b ∨ b + 4 * (edge e : ℤ) < 2 * (gap f e : ℤ)
      ∨ 2 * (gap f e : ℤ) + 4 * (edge e : ℤ) < b := by
  by_contra hcon
  push Not at hcon
  obtain ⟨hne, hhi, hlo⟩ := hcon
  obtain ⟨q, hcert⟩ := exp_windows_refuted e hr.range
  rcases rest_eq f e hr with hrest | ⟨heq, -⟩
  · rcases lt_or_ge (2 * (gap f e : ℤ)) b with h | h
    · exact no_window_hit f e hr hcert hbelow (by rw [hrest]; omega)
        (by rw [hrest]; omega)
    · exact no_window_hit f e hr hcert habove (by rw [hrest]; omega)
        (by rw [hrest]; omega)
  · rw [heq] at hhi
    omega

/-- The trim-down dichotomy, about the boundary `num`. -/
theorem down_tie_or_far (f : ℕ) (e : ℤ) (hr : binary64.Regular f e) :
    2 * (gap f e : ℤ) = num e
      ∨ (num e : ℤ) + 4 * edge e < 2 * (gap f e : ℤ)
      ∨ 2 * (gap f e : ℤ) + 4 * (edge e : ℤ) < num e := by
  have hb := edge_bounds e hr.range
  exact gap_tie_or_far (b := (num e : ℤ)) f e hr
    ⟨by exact_mod_cast hb.1, by exact_mod_cast hb.2⟩
    (by simp [expWindows, Format.regularWindows])
    (by simp [expWindows, Format.regularWindows])

/-- The trim-up dichotomy, about the boundary `2·step - num`, which Żmij's carry
    reads as the fraction and the half ULP summing past `2^64`. -/
theorem up_tie_or_far (f : ℕ) (e : ℤ) (hr : binary64.Regular f e) :
    2 * (gap f e : ℤ) + num e = 2 * (step e : ℤ)
      ∨ 2 * (step e : ℤ) + 4 * edge e < 2 * (gap f e : ℤ) + num e
      ∨ 2 * (gap f e : ℤ) + (num e : ℤ) + 4 * edge e < 2 * step e := by
  have hb := edge_bounds e hr.range
  have hnum := num_pos e hr.range
  have hbz : 4 * (edge e : ℤ) < 2 * (step e : ℤ) - num e
      ∧ 2 * (step e : ℤ) - num e + 4 * (edge e : ℤ) < 2 * (step e : ℤ) := by
    obtain ⟨h1, h2⟩ := hb
    constructor
    · have : (num e : ℤ) + 4 * edge e < 2 * step e := by exact_mod_cast h2
      omega
    · have : 4 * (edge e : ℤ) < num e := by exact_mod_cast h1
      omega
  have := gap_tie_or_far (b := 2 * (step e : ℤ) - num e) f e hr hbz
    (by simp [expWindows, Format.regularWindows])
    (by simp [expWindows, Format.regularWindows])
  omega

/-! ## The coarse decisions

Each flag is pinned to its candidate by two composed equivalences: `flag ↔ gap`
says what the comparison decides about `gap`, and `roundtrips_iff_dist` says
when that gap admits a round-trip.

The first is where the work is. One unit of the fraction is worth one `edge` of
the gap, so the comparison is the exact one up to the two truncation errors, and
the dichotomies above leave only the exact ties inside that margin. There both
sides resolve the same way, the flag by the `+1` for even `f` and the
specification by the parity in `roundtrips_iff_dist`.
-/

/-- Units of the fraction are worth that many `edge` of the gap. -/
private theorem edge_add_le (e : ℤ) {a b n : ℕ} (h : a + n ≤ b) :
    edge e * a + n * edge e ≤ edge e * b := by
  calc edge e * a + n * edge e = edge e * (a + n) := by ring
    _ ≤ edge e * b := Nat.mul_le_mul_left _ h

/-- Twice the fraction's unit as a power of two, which is what `half`
    truncates by. -/
private theorem two_unit_eq (e : ℤ) (hs : exponentShift e ≤ 9) :
    2 * unit e = 2 ^ (74 - exponentShift e) := by
  rw [unit, show 74 - exponentShift e = 73 - exponentShift e + 1 from by omega,
    pow_succ]
  ring

/-- The truncated half ULP is positive and below `2^64`: the power of ten is
    normalized and the shift truncates it by between `2^65` and `2^68`. -/
theorem half_bounds (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    0 < half e ∧ half e < 2 ^ 64 := by
  obtain ⟨hs6, hs9⟩ := exponent_shift_range e he
  obtain ⟨hn, hn'⟩ := num_bounds e he
  obtain ⟨hlo, hhi⟩ := binary64.power10_significand_bounds hn hn'
  have hlo' : (2 : ℕ) ^ 127 ≤ p10 e := hlo
  have hhi' : p10 e < 2 ^ 128 := hhi
  have htwo := two_unit_eq e hs9
  have hpos : 0 < 2 * unit e := by have := unit_pos e; omega
  have hsmall : 2 * unit e ≤ 2 ^ 68 := by
    rw [htwo]; exact Nat.pow_le_pow_right (by norm_num) (by omega)
  have hbig : (2 : ℕ) ^ 65 ≤ 2 * unit e := by
    rw [htwo]; exact Nat.pow_le_pow_right (by norm_num) (by omega)
  refine ⟨?_, ?_⟩
  · have h1 : 1 ≤ p10 e / (2 * unit e) := (Nat.one_le_div_iff hpos).mpr
      (calc 2 * unit e ≤ 2 ^ 68 := hsmall
        _ ≤ 2 ^ 127 := Nat.pow_le_pow_right (by norm_num) (by norm_num)
        _ ≤ p10 e := hlo')
    rw [half]
    omega
  · rw [half, Nat.div_lt_iff_lt_mul hpos]
    calc p10 e < 2 ^ 128 := hhi'
      _ = 2 ^ 63 * 2 ^ 65 := by norm_num
      _ ≤ 2 ^ 64 * 2 ^ 65 := by norm_num
      _ ≤ 2 ^ 64 * (2 * unit e) := Nat.mul_le_mul_left _ hbig

/-- The fraction and the half ULP never sum to exactly `2^64`. Where they
    did, the dichotomy would make the value an exact tie with both truncations
    vanishing, and the power of ten being exact then forces
    `(2f+1)·half = (c+1)·2^64`; but `2f+1` is odd and `half` is positive and
    below `2^64`, so no significand can do it. This is what keeps Żmij's carry
    from firing on an odd significand at a tie, where the upper candidate does
    not round-trip. -/
theorem fraction_add_half_ne (f : ℕ) (e : ℤ) (hr : binary64.Regular f e) :
    fractionalPart f e + half e ≠ 2 ^ 64 := by
  intro hS
  have hs := (exponent_shift_range e hr.range).2
  have hden := den_pos e
  have hedge := edge_pos e
  have hgap := gap_eq_edge_fraction f e hs
  have hnum := num_eq_edge_half e
  obtain ⟨-, herr2⟩ := err_bounds f e hr
  -- The doubled gap is within a fraction unit of the upper boundary, so the
  -- dichotomy makes it an exact tie and both truncations vanish.
  have hmul : edge e * fractionalPart f e + edge e * half e = step e := by
    rw [show edge e * fractionalPart f e + edge e * half e
        = edge e * (fractionalPart f e + half e) from by ring, hS, step_eq_edge]
  have hzero : 2 * err f e + errHalf e = 0 := by
    rcases up_tie_or_far f e hr with h | h | h <;> omega
  -- An exact power of ten with nothing below the half ULP, so one ULP is
  -- exactly twice it and the gap is exactly the fraction.
  have hτ : num e % den e = 0 := by rw [errHalf] at hzero; omega
  have hsig : p10 e % (2 * unit e) = 0 := by
    rw [errHalf] at hzero
    rcases Nat.eq_zero_or_pos (p10 e % (2 * unit e)) with h | h
    · exact h
    · have : 0 < den e * (p10 e % (2 * unit e)) := Nat.mul_pos hden h
      omega
  have hnum' : num e = 2 * (edge e * half e) := by rw [num_eq_edge_half]; omega
  have hgap' : gap f e = edge e * fractionalPart f e := by omega
  -- Scaling the exact identity back up and cancelling `edge` leaves an odd
  -- multiple of `half` equal to a multiple of `2^64`.
  have hid := integral_add_gap f e hs
  have hkey : edge e * ((integralPart f e + 1) * 2 ^ 64)
      = edge e * ((2 * f + 1) * half e) := by
    have hstep : integralPart f e * step e = edge e * (integralPart f e * 2 ^ 64) := by
      rw [step_eq_edge]; ring
    have hrhs : f * num e = edge e * (2 * f * half e) := by rw [hnum']; ring
    have hlhs : edge e * fractionalPart f e + edge e * half e
        = edge e * 2 ^ 64 := by rw [hmul, step_eq_edge]
    calc edge e * ((integralPart f e + 1) * 2 ^ 64)
        = edge e * (integralPart f e * 2 ^ 64) + edge e * 2 ^ 64 := by ring
      _ = integralPart f e * step e
            + (edge e * fractionalPart f e + edge e * half e) := by
          rw [hstep, hlhs]
      _ = f * num e + edge e * half e := by rw [← hgap', ← hid]; ring
      _ = edge e * ((2 * f + 1) * half e) := by rw [hrhs]; ring
  have hcancel : (integralPart f e + 1) * 2 ^ 64 = (2 * f + 1) * half e :=
    Nat.eq_of_mul_eq_mul_left hedge hkey
  -- `2f+1` is odd, so the power of two has to divide `half`, which is too
  -- small to admit it.
  have hcop : Nat.Coprime (2 ^ 64) (2 * f + 1) :=
    Nat.Coprime.pow_left _ (Nat.coprime_two_left.mpr ⟨f, by ring⟩)
  have hdvd : (2 : ℕ) ^ 64 ∣ half e := by
    refine hcop.dvd_of_dvd_mul_right ⟨integralPart f e + 1, ?_⟩
    calc half e * (2 * f + 1) = (2 * f + 1) * half e := by ring
      _ = (integralPart f e + 1) * 2 ^ 64 := hcancel.symm
      _ = 2 ^ 64 * (integralPart f e + 1) := by ring
  obtain ⟨hpos, hlt⟩ := half_bounds e hr.range
  exact absurd (Nat.le_of_dvd hpos hdvd) (by omega)

/-- What `roundDown` decides. The comparison is the exact bound on `gap` up to
    the two truncations, and inside that margin `down_tie_or_far` leaves only an
    exact tie, where the fraction equals the half ULP exactly and the `+1`
    for even `f` resolves it the way the specification does. -/
theorem round_down_iff_gap (f : ℕ) (e : ℤ) (hr : binary64.Regular f e) :
    (toDecimalCandidates f e).roundDown = true
      ↔ if f % 2 = 0 then 2 * gap f e ≤ num e else 2 * gap f e < num e := by
  have hs := (exponent_shift_range e hr.range).2
  have hgap := gap_eq_edge_fraction f e hs
  have hnum := num_eq_edge_half e
  obtain ⟨herr, -⟩ := err_bounds f e hr
  have hhalf := err_half_lt e
  have hdich := down_tie_or_far f e hr
  have hflag : (toDecimalCandidates f e).roundDown = true
      ↔ fractionalPart f e < half e + (1 - f % 2) := by
    show decide (fractionalPart f e < halfUlp f e) = true ↔ _
    rw [decide_eq_true_eq, half_ulp_eq f e hs]
  rw [hflag]
  rcases Nat.lt_trichotomy (fractionalPart f e) (half e) with hf | hf | hf
  · have hp := edge_add_le e (by omega : fractionalPart f e + 1 ≤ half e)
    split_ifs <;> omega
  · have hp : edge e * fractionalPart f e = edge e * half e := by rw [hf]
    split_ifs <;> omega
  · have hp := edge_add_le e (by omega : half e + 1 ≤ fractionalPart f e)
    split_ifs <;> omega

/-- What `roundUp` decides. Żmij detects it as the carry out of a 64-bit
    addition, which is the fraction and the half ULP summing past `2^64`;
    `fraction_add_half_ne` rules out landing on `2^64` itself, and one short of
    it `up_tie_or_far` again leaves only an exact tie. -/
theorem round_up_iff_gap (f : ℕ) (e : ℤ) (hr : binary64.Regular f e) :
    (toDecimalCandidates f e).roundUp = true
      ↔ if f % 2 = 0 then 2 * step e ≤ 2 * gap f e + num e
        else 2 * step e < 2 * gap f e + num e := by
  have hs := (exponent_shift_range e hr.range).2
  have hgap := gap_eq_edge_fraction f e hs
  have hnum := num_eq_edge_half e
  obtain ⟨-, herr2⟩ := err_bounds f e hr
  have hdich := up_tie_or_far f e hr
  have hne := fraction_add_half_ne f e hr
  have hmul := step_eq_edge e
  have hsum : edge e * fractionalPart f e + edge e * half e
      = edge e * (fractionalPart f e + half e) := by ring
  have hflag : (toDecimalCandidates f e).roundUp = true
      ↔ 2 ^ 64 ≤ fractionalPart f e + (half e + (1 - f % 2)) := by
    show decide (2 ^ 64 ≤ fractionalPart f e + halfUlp f e) = true ↔ _
    rw [decide_eq_true_eq, half_ulp_eq f e hs]
  rw [hflag]
  rcases Nat.lt_trichotomy (fractionalPart f e + half e) (2 ^ 64) with hf | hf | hf
  · -- Short of the carry: either two units short, or one short and a tie.
    rcases Nat.lt_or_ge (fractionalPart f e + half e + 1) (2 ^ 64) with h1 | h1
    · have hp := edge_add_le e (by omega : fractionalPart f e + half e + 2 ≤ 2 ^ 64)
      split_ifs <;> omega
    · have heq : fractionalPart f e + half e + 1 = 2 ^ 64 := by omega
      have hp : edge e * (fractionalPart f e + half e) + edge e
          = edge e * 2 ^ 64 := by
        calc edge e * (fractionalPart f e + half e) + edge e
            = edge e * (fractionalPart f e + half e + 1) := by ring
          _ = edge e * 2 ^ 64 := by rw [heq]
      split_ifs <;> omega
  · exact absurd hf hne
  · have hp := edge_add_le e
      (by omega : 2 ^ 64 + 1 ≤ fractionalPart f e + half e)
    split_ifs <;> omega

/-- The integral part sits `gap` below the scaled value. -/
theorem integral_scaled (f : ℕ) (e : ℤ) (hs : exponentShift e ≤ 9) :
    (integralPart f e : ℤ) * step e + gap f e = f * num e := by
  exact_mod_cast integral_add_gap f e hs

/-- And its successor one whole step above that. -/
theorem successor_scaled (f : ℕ) (e : ℤ) (hs : exponentShift e ≤ 9) :
    ((integralPart f e + 1 : ℕ) : ℤ) * step e + ((gap f e : ℤ) - step e)
      = f * num e := by
  have := integral_scaled f e hs
  push_cast
  linarith

/-- `roundDown` fires exactly when the integral part round-trips. Both are a
    bound on `gap` by `num`, the gap being that candidate's distance from the
    scaled value, and the lower end of the round-trip interval is free because
    a gap is never negative. -/
theorem round_down_iff_roundtrips (f : ℕ) (e : ℤ) (hr : binary64.Regular f e) :
    (toDecimalCandidates f e).roundDown = true
      ↔ Roundtrips f e
          (integralPart f e * 10 ^ (binary64.decimalExponent e + 1)) := by
  rw [round_down_iff_gap f e hr,
    roundtrips_iff_dist f e hr.range
      (integral_scaled f e (exponent_shift_range e hr.range).2)]
  split_ifs <;> omega

/-- `roundUp` fires exactly when the successor round-trips. Its distance is
    signed, and the upper end of the interval is free because the gap never
    passes a whole step. -/
theorem round_up_iff_roundtrips (f : ℕ) (e : ℤ) (hr : binary64.Regular f e) :
    (toDecimalCandidates f e).roundUp = true
      ↔ Roundtrips f e
          ((integralPart f e + 1 : ℕ)
            * 10 ^ (binary64.decimalExponent e + 1)) := by
  have hle := gap_le_step f e hr
  have hnum := num_pos e hr.range
  rw [round_up_iff_gap f e hr,
    roundtrips_iff_dist f e hr.range
      (successor_scaled f e (exponent_shift_range e hr.range).2)]
  split_ifs <;> omega

/-! ## The derived digit

Ten times the fraction, rounded, is Żmij's last digit, and the `+6` in
`biasedHalf` is the rest of the error budget: it stands in for the part of the
gap the fraction dropped, which is worth up to ten of the units the digit
is decided in. So the digit is a nearest one except within ten units of a digit
boundary, where a `+6` can be wrong by six either way.

What closes those cases is how rigid they are. Reaching a boundary asks twenty
times the fraction to come within twenty of a multiple of `2^64`, and since the
fraction also fixes the digit, that leaves eleven possible fractions in all,
each an explicit constant. Each one fixes how much of the truncation the `+6`
would have to have got right, which is a bound on the remainder below one
fraction unit, and the certificates below say no significand meets it. Of these
eleven exceptional fractions, only the exact midpoint at a quarter survives,
where the truncation vanishes and the digit is two: that is the tie, and Żmij's
special case rounds it to even.
-/

/-! ### The digit error

The digit's distance from the value splits the way `res_split` splits the
residue: into the fraction, which the digit is computed from, and the
remainder below one fraction unit, which the `+6` stands in for. That remainder
is what the boundary cases below turn on, so it is bounded first, at twenty of
it under twenty-one units, the twenty being the ten of the base-ten multiply
doubled.
-/

/-- Twice the signed distance from Żmij's fine output to the exact value, on the
    grid at `k` and multiplied through by `step`: the digit accounts for ten
    times the gap, and this is what it leaves. -/
def digitDist (f : ℕ) (e : ℤ) : ℤ :=
  20 * (gap f e : ℤ) - 2 * (toDecimalCandidates f e).digit * step e

/-- The truncation stays under one fraction unit with room to spare: the
    remainder below the unit accounts for nearly all of it and the power-of-ten
    truncation for at most `2^53` denominators, which one unit dwarfs. -/
theorem err_lt (f : ℕ) (e : ℤ) (hr : binary64.Regular f e) :
    20 * err f e < 21 * edge e := by
  have hden := den_pos e
  have hunit : 20 * 2 ^ 53 ≤ unit e := by
    have hs := (exponent_shift_range e hr.range).2
    rw [unit]
    calc 20 * 2 ^ 53 ≤ 2 ^ 64 := by norm_num
      _ ≤ 2 ^ (73 - exponentShift e) :=
          Nat.pow_le_pow_right (by norm_num) (by omega)
  -- The remainder below one unit, in integers, is one unit short of the whole
  -- edge.
  have hrest : den e * (f * p10 e % unit e) + den e ≤ edge e := by
    have h : f * p10 e % unit e + 1 ≤ unit e := by
      have := Nat.mod_lt (f * p10 e) (unit_pos e)
      omega
    rw [edge]
    calc den e * (f * p10 e % unit e) + den e
        = den e * (f * p10 e % unit e + 1) := by ring
      _ ≤ den e * unit e := Nat.mul_le_mul (le_refl _) h
      _ = unit e * den e := by ring
  -- The power-of-ten truncation is smaller still, by a whole unit.
  have htau : 20 * (f * (num e % den e)) + unit e ≤ edge e := by
    have h1 : 20 * f ≤ unit e := by have : f < 2 ^ 53 := hr.sig_lt; omega
    have h2 : num e % den e ≤ den e - 1 := by
      have := Nat.mod_lt (num e) hden
      omega
    have hsucc : den e - 1 + 1 = den e := by omega
    rw [edge]
    calc 20 * (f * (num e % den e)) + unit e
        = 20 * f * (num e % den e) + unit e := by ring
      _ ≤ unit e * (den e - 1) + unit e :=
          Nat.add_le_add_right (Nat.mul_le_mul h1 h2) _
      _ = unit e * (den e - 1 + 1) := by ring
      _ = unit e * den e := by rw [hsucc]
  have := unit_pos e
  rw [err]
  omega

/-- A bound on the truncation is a bound on the remainder below one unit: the
    denominator divides out of both. -/
private theorem res_le_of_err (f : ℕ) (e : ℤ) {a : ℕ}
    (h : 20 * err f e ≤ a * edge e) :
    20 * (f * p10 e % unit e) ≤ a * unit e := by
  refine Nat.le_of_mul_le_mul_left ?_ (den_pos e)
  calc den e * (20 * (f * p10 e % unit e))
      = 20 * (den e * (f * p10 e % unit e)) := by ring
    _ ≤ 20 * err f e := by rw [err]; omega
    _ ≤ a * edge e := h
    _ = den e * (a * unit e) := by rw [edge]; ring

/-- And conversely, up to what the power-of-ten truncation can contribute. -/
private theorem res_ge_of_err (f : ℕ) (e : ℤ)
    (hr : binary64.Regular f e) {a : ℕ}
    (h : a * edge e ≤ 20 * err f e) :
    a * unit e ≤ 20 * (f * p10 e % unit e) + 20 * 2 ^ 53 := by
  have htau : f * (num e % den e) ≤ 2 ^ 53 * den e := by
    have h1 : f ≤ 2 ^ 53 := by have : f < 2 ^ 53 := hr.sig_lt; omega
    have h2 : num e % den e ≤ den e := le_of_lt (Nat.mod_lt _ (den_pos e))
    exact Nat.mul_le_mul h1 h2
  refine Nat.le_of_mul_le_mul_left ?_ (den_pos e)
  calc den e * (a * unit e) = a * edge e := by rw [edge]; ring
    _ ≤ 20 * err f e := h
    _ ≤ den e * (20 * (f * p10 e % unit e) + 20 * 2 ^ 53) := by
        rw [err]
        calc 20 * (den e * (f * p10 e % unit e) + f * (num e % den e))
            = den e * (20 * (f * p10 e % unit e))
              + 20 * (f * (num e % den e)) := by ring
          _ ≤ den e * (20 * (f * p10 e % unit e))
              + 20 * (2 ^ 53 * den e) := by omega
          _ = den e * (20 * (f * p10 e % unit e) + 20 * 2 ^ 53) := by ring

/-! ### Refuting the digit windows

Each of the eleven fractions that can reach a digit boundary asks the
truncation to reach a definite distance, and each such demand is a window of
residues of `f·p10` modulo one coarse step. The generic machinery of
`ModWindows` refutes every one of those windows over the exponent range, one
exponent at a time.

Do the symbolic case reduction before reaching for a certificate. A wider
formulation that enumerates every potentially bad input looks simpler, but
preserving the algebraic structure of each decision boundary is what turns a
nonempty exceptional set into an empty certificate problem. Stated instead as
one band around the digit boundary, the ambiguity has occupants at six
exponents, and each would have to be named and discharged separately.
-/

/-- The six fractions that can reach a digit boundary from below, each with
    how far the truncation would have to reach for it to happen, in fifths of a
    fraction unit. -/
private def digitLowEdges : List (ℕ × ℕ) :=
  [(8301034833169298227, 1), (17524406870024074035, 1),
    (2767011611056432742, 2), (11990383647911208550, 2),
    (6456360425798343065, 3), (15679732462653118873, 3)]

/-- And the five that can reach one from above. The first is the quarter Żmij
    special-cases, where the reach needed is nothing at all: only the exact
    midpoint survives, so only a vanishing truncation does. -/
private def digitHighEdges : List (ℕ × ℕ) :=
  [(4611686018427387904, 0), (922337203685477580, 4),
    (10145709240540253388, 4), (4611686018427387903, 5),
    (13835058055282163711, 5)]

private def digitLowWindows (e : ℤ) : ModWindows :=
  binary64.regularWindows (p10 e) (unit e * 2 ^ 64) e <|
    digitLowEdges.map fun p =>
      ((unit e : ℤ) * p.1, (unit e : ℤ) * p.1 + p.2 * (unit e : ℤ) / 5)

private def digitHighWindows (e : ℤ) : ModWindows :=
  binary64.regularWindows (p10 e) (unit e * 2 ^ 64) e <|
    digitHighEdges.map fun p =>
      ((unit e : ℤ) * p.1
          + max (p.2 * (unit e : ℤ) / 5 - 2 ^ 53)
              (if num e % den e = 0 then 1 else 0),
        (unit e : ℤ) * p.1 + unit e - 1)

/-- Close the two digit certificates for a literal exponent. -/
elab "digit_low_cert" : tactic => modCertTactic fun e => (digitLowWindows e).search

elab "digit_high_cert" : tactic => modCertTactic fun e => (digitHighWindows e).search

private theorem digit_low_refuted (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    ∃ q, (digitLowWindows e).refutedBy q = true := by
  obtain ⟨hlo, hhi⟩ := he
  interval_cases e <;> digit_low_cert

private theorem digit_high_refuted (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    ∃ q, (digitHighWindows e).refutedBy q = true := by
  obtain ⟨hlo, hhi⟩ := he
  interval_cases e <;> digit_high_cert

/-- A fraction together with a bound on the remainder below the unit is a
    point of `res`, which is the residue of `f·p10` modulo one coarse step, so
    a refuted window rules the pair out. -/
private theorem no_res_hit {rmin rmax q : ℤ} {windows : List (ℤ × ℤ)} (f : ℕ)
    (e : ℤ) (hr : binary64.Regular f e)
    (hcert : (binary64.regularWindows (p10 e) (unit e * 2 ^ 64) e
      windows).refutedBy q = true)
    (hmem : (rmin, rmax) ∈ windows)
    (hrmin : rmin ≤ (res f e : ℤ)) (hrmax : (res f e : ℤ) ≤ rmax) : False :=
  binary64.regular_not_hit f hr (by have := unit_pos e; positivity) hcert hmem
    (by rw [res, Nat.mul_comm]) hrmin hrmax

/-- No significand pairs one of the six low fractions with a truncation
    that reaches the boundary. -/
private theorem no_digit_low (f : ℕ) (e : ℤ)
    (hr : binary64.Regular f e) {w j : ℕ}
    (hmem : (w, j) ∈ digitLowEdges) (hφ : fractionalPart f e = w)
    (hb : 20 * (f * p10 e % unit e) ≤ 4 * j * unit e) : False := by
  obtain ⟨q, hcert⟩ := digit_low_refuted e hr.range
  have hres : (res f e : ℤ)
      = (unit e : ℤ) * w + (f * p10 e % unit e : ℕ) := by
    rw [res_split f e (exponent_shift_range e hr.range).2, hφ]
    push_cast
    ring
  have hdiv : ((f * p10 e % unit e : ℕ) : ℤ) ≤ (j : ℤ) * (unit e : ℤ) / 5 := by
    refine Int.le_ediv_of_mul_le (by norm_num) ?_
    have : (20 : ℤ) * (f * p10 e % unit e : ℕ) ≤ 4 * j * (unit e : ℤ) := by
      exact_mod_cast hb
    linarith
  have hmem' : ((unit e : ℤ) * w, (unit e : ℤ) * w + (j : ℤ) * (unit e : ℤ) / 5)
      ∈ (digitLowWindows e).windows :=
    List.mem_map_of_mem (l := digitLowEdges) (f := fun p : ℕ × ℕ =>
      ((unit e : ℤ) * p.1, (unit e : ℤ) * p.1 + p.2 * (unit e : ℤ) / 5)) hmem
  exact no_res_hit f e hr hcert hmem'
    (by rw [hres]; omega) (by rw [hres]; omega)

/-- Nor with one of the five high fractions, the quarter included: there the
    truncation has to vanish, and it does not. -/
private theorem no_digit_high (f : ℕ) (e : ℤ)
    (hr : binary64.Regular f e) {w j : ℕ}
    (hmem : (w, j) ∈ digitHighEdges) (hφ : fractionalPart f e = w)
    (hb : 4 * j * unit e ≤ 20 * (f * p10 e % unit e) + 20 * 2 ^ 53)
    (hpos : 0 < err f e) : False := by
  obtain ⟨q, hcert⟩ := digit_high_refuted e hr.range
  have hres : (res f e : ℤ)
      = (unit e : ℤ) * w + (f * p10 e % unit e : ℕ) := by
    rw [res_split f e (exponent_shift_range e hr.range).2, hφ]
    push_cast
    ring
  have hunit : (0 : ℤ) < unit e := by exact_mod_cast unit_pos e
  have hdiv : (j : ℤ) * (unit e : ℤ) / 5 - 2 ^ 53
      ≤ ((f * p10 e % unit e : ℕ) : ℤ) := by
    have hb' : 4 * (j : ℤ) * (unit e : ℤ)
        ≤ 20 * (f * p10 e % unit e : ℕ) + 20 * 2 ^ 53 := by exact_mod_cast hb
    have := Int.ediv_le_ediv (by norm_num : (0 : ℤ) < 5)
      (show (j : ℤ) * (unit e : ℤ)
        ≤ 5 * (((f * p10 e % unit e : ℕ) : ℤ) + 2 ^ 53) from by linarith)
    rw [Int.mul_ediv_cancel_left _ (by norm_num)] at this
    omega
  -- The truncation is positive, so the remainder is too unless the power of ten
  -- is exact, which is the only case the window's own lower end has to cover.
  have hone : (if num e % den e = 0 then (1 : ℤ) else 0)
      ≤ ((f * p10 e % unit e : ℕ) : ℤ) := by
    split_ifs with hτ
    · rcases Nat.eq_zero_or_pos (f * p10 e % unit e) with hz | hz
      · rw [err, hτ, hz] at hpos
        simp at hpos
      · exact_mod_cast hz
    · positivity
  have hlt : ((f * p10 e % unit e : ℕ) : ℤ) ≤ (unit e : ℤ) - 1 := by
    have := Nat.mod_lt (f * p10 e) (unit_pos e)
    omega
  have hmem' : ((unit e : ℤ) * w
        + max ((j : ℤ) * (unit e : ℤ) / 5 - 2 ^ 53)
            (if num e % den e = 0 then 1 else 0),
      (unit e : ℤ) * w + unit e - 1) ∈ (digitHighWindows e).windows :=
    List.mem_map_of_mem (l := digitHighEdges) (f := fun p : ℕ × ℕ =>
      ((unit e : ℤ) * p.1
          + max (p.2 * (unit e : ℤ) / 5 - 2 ^ 53)
              (if num e % den e = 0 then 1 else 0),
        (unit e : ℤ) * p.1 + unit e - 1)) hmem
  exact no_res_hit f e hr hcert hmem'
    (by rw [hres]
        rcases max_cases ((j : ℤ) * (unit e : ℤ) / 5 - 2 ^ 53)
            (if num e % den e = 0 then (1 : ℤ) else 0) with ⟨h, -⟩ | ⟨h, -⟩ <;>
          rw [h] <;> omega)
    (by rw [hres]; omega)

/-! ### The table

Żmij reads the digit from `digitOffset` rather than computing `roundedDigit`.
The table places the boundaries after the digits 0, 1, 2, 8 and 9 where
`(frac·10 + 2^63 - 1) / 2^64` steps up, which rounds the quarter's 2.5 to even
by itself, and those after 3 to 7 where `(frac·10 + 2^63 + 7) / 2^64` does.
The second rule moves no boundary of `roundedDigit`, but the first moves three
up by one fraction unit, those after 1, 8 and 9, so the two digits differ at
three fractions. All three are low edges, which the certificates of
`digit_low_refuted` exclude only with a small truncation. Here the whole cell of
each is refuted, which is what lets `table_digit_eq` state the two digits are
equal.
-/

/-- The cell read from the high word is the top four bits of the fraction. -/
theorem cell_eq (f : ℕ) (e : ℤ) :
    scaledSignificand f e / 2 ^ 69 % 16 = fractionalPart f e / 2 ^ 60 := by
  rw [fractionalPart]
  omega

/-- The table's digit is `roundedDigit` at every fraction but three. -/
private theorem table_digit_eq_rounded {φ : ℕ} (hφ : φ < 2 ^ 64)
    (h12 : φ ≠ 2767011611056432742) (h89 : φ ≠ 15679732462653118873)
    (h910 : φ ≠ 17524406870024074035) :
    (φ + digitOffset (φ / 2 ^ 60)) % 2 ^ 64 / 2 ^ 60 = roundedDigit φ := by
  obtain ⟨c, hc⟩ : ∃ c, φ / 2 ^ 60 = c := ⟨_, rfl⟩
  have hc16 : c < 16 := by omega
  -- Name the reference digit and split on the carry out of the table's sum:
  -- with both fixed, every comparison left is linear in `φ`, which `omega`
  -- needs, having no dark shadow to reason across two divisions.
  obtain ⟨d, hd⟩ : ∃ d, (φ * 10 + biasedHalf) / 2 ^ 64 = d := ⟨_, rfl⟩
  have hd10 : d ≤ 10 := by rw [biasedHalf] at hd; omega
  have hK : digitOffset c < 2 ^ 64 := by
    interval_cases c <;> simp only [digitOffset] <;> omega
  rw [hc, roundedDigit, hd]
  rw [biasedHalf] at hd
  rcases Nat.lt_or_ge (φ + digitOffset c) (2 ^ 64) with hcarry | hcarry
  · rw [Nat.mod_eq_of_lt hcarry]
    interval_cases c <;> simp only [digitOffset] at hcarry ⊢ <;>
      interval_cases d <;> split_ifs <;> omega
  · rw [Nat.mod_eq_sub_mod hcarry, Nat.mod_eq_of_lt (by omega)]
    interval_cases c <;> simp only [digitOffset] at hcarry ⊢ <;>
      interval_cases d <;> split_ifs <;> omega

/-- The cell of a changed fraction: every remainder below its unit. -/
private def cellWindows (w : ℕ) (e : ℤ) : ModWindows :=
  binary64.regularWindows (p10 e) (unit e * 2 ^ 64) e
    [((unit e : ℤ) * w, (unit e : ℤ) * w + unit e - 1)]

/-- Close a cell certificate for a literal fraction and exponent. -/
elab "cell_cert " w:num : tactic =>
  modCertTactic fun e => (cellWindows w.getNat e).search

/-- The fraction the table moves from digit two down to one. -/
private theorem cell_12_refuted (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    ∃ q, (cellWindows 2767011611056432742 e).refutedBy q = true := by
  obtain ⟨hlo, hhi⟩ := he
  interval_cases e <;> cell_cert 2767011611056432742

/-- From nine down to eight. -/
private theorem cell_89_refuted (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    ∃ q, (cellWindows 15679732462653118873 e).refutedBy q = true := by
  obtain ⟨hlo, hhi⟩ := he
  interval_cases e <;> cell_cert 15679732462653118873

/-- From ten down to nine. -/
private theorem cell_910_refuted (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    ∃ q, (cellWindows 17524406870024074035 e).refutedBy q = true := by
  obtain ⟨hlo, hhi⟩ := he
  interval_cases e <;> cell_cert 17524406870024074035

/-- No significand has a fraction whose cell is refuted. -/
private theorem no_cell_hit (f : ℕ) (e : ℤ) (hr : binary64.Regular f e) {w : ℕ}
    (hcert : ∃ q, (cellWindows w e).refutedBy q = true)
    (hφ : fractionalPart f e = w) : False := by
  obtain ⟨q, hcert⟩ := hcert
  have hres : (res f e : ℤ)
      = (unit e : ℤ) * w + (f * p10 e % unit e : ℕ) := by
    rw [res_split f e (exponent_shift_range e hr.range).2, hφ]
    push_cast
    ring
  have hlt : ((f * p10 e % unit e : ℕ) : ℤ) ≤ (unit e : ℤ) - 1 := by
    have := Nat.mod_lt (f * p10 e) (unit_pos e)
    omega
  exact no_res_hit f e hr hcert (List.mem_singleton_self _)
    (by rw [hres]; omega) (by rw [hres]; omega)

/-- Żmij's table digit is `roundedDigit` for every regular input: the three
    fractions where the two differ are never reached. -/
theorem table_digit_eq (f : ℕ) (e : ℤ) (hr : binary64.Regular f e) :
    (toDecimalCandidates f e).digit = roundedDigit (fractionalPart f e) := by
  have hφ : fractionalPart f e < 2 ^ 64 := by
    rw [fractionalPart]
    exact Nat.mod_lt _ (by norm_num)
  have hdigit : (toDecimalCandidates f e).digit
      = (fractionalPart f e
          + digitOffset (scaledSignificand f e / 2 ^ 69 % 16)) % 2 ^ 64
        / 2 ^ 60 := rfl
  rw [hdigit, cell_eq]
  exact table_digit_eq_rounded hφ
    (fun h => no_cell_hit f e hr (cell_12_refuted e hr.range) h)
    (fun h => no_cell_hit f e hr (cell_89_refuted e hr.range) h)
    (fun h => no_cell_hit f e hr (cell_910_refuted e hr.range) h)

/-! ### Nearest at the digit

What is left is finite arithmetic on the fraction and the digit it
produces. The biased quotient bounds how far the two can be out of step;
reaching either boundary pins the fraction to an entry of the tables above,
where the certificates rule it out; and the one exact midpoint Żmij does not
special-case has an even digit. `digit_nearest` assembles these.
-/

/-- The biased quotient bounds the offset of twenty times the fraction from
    the digit boundary below it: the `+6` is worth twelve of those units, so the
    offset can fall short of that boundary by at most twelve and of the one
    above by at least thirteen. -/
private theorem digit_offset_bounds {φ d m : ℕ} {v : ℤ} (hφ : φ < 2 ^ 64)
    (hfloor : 2 ^ 64 * d + m = φ * 10 + (2 ^ 63 + 6)) (hm : m < 2 ^ 64)
    (hv : v = 20 * φ + 2 ^ 64 - 2 * d * 2 ^ 64) :
    d ≤ 10 ∧ -12 ≤ v ∧ v ≤ 2 ^ 65 - 13 := by
  omega

/-- Falling short of the boundary below makes the offset a negative multiple of
    four, and since the fraction fixes the digit too, each of its three
    values leaves two possible fractions. -/
private theorem digit_low_cases {φ d : ℕ} {v : ℤ} (hφ : φ < 2 ^ 64)
    (hd : d ≤ 10) (hv : v = 20 * φ + 2 ^ 64 - 2 * d * 2 ^ 64)
    (hlo : -12 ≤ v) (hhi : v < 0) :
    ∃ w j : ℕ, (w, j) ∈ digitLowEdges ∧ (φ : ℤ) = w ∧ v = -(4 * (j : ℤ)) := by
  interval_cases d <;>
    first
      | omega
      | exact ⟨8301034833169298227, 1, by decide, by omega, by omega⟩
      | exact ⟨17524406870024074035, 1, by decide, by omega, by omega⟩
      | exact ⟨2767011611056432742, 2, by decide, by omega, by omega⟩
      | exact ⟨11990383647911208550, 2, by decide, by omega, by omega⟩
      | exact ⟨6456360425798343065, 3, by decide, by omega, by omega⟩
      | exact ⟨15679732462653118873, 3, by decide, by omega, by omega⟩

/-- Reaching past the boundary above leaves four, the quarter aside. -/
private theorem digit_high_cases {φ d : ℕ} {v : ℤ} (hφ : φ < 2 ^ 64)
    (hd : d ≤ 10) (hv : v = 20 * φ + 2 ^ 64 - 2 * d * 2 ^ 64)
    (hlo : 2 ^ 65 - 20 ≤ v) (hhi : v ≤ 2 ^ 65 - 13) :
    ∃ w j : ℕ, (w, j) ∈ digitHighEdges ∧ (φ : ℤ) = w
      ∧ v = 2 ^ 65 - 4 * (j : ℤ) := by
  interval_cases d <;>
    first
      | omega
      | exact ⟨922337203685477580, 4, by decide, by omega, by omega⟩
      | exact ⟨10145709240540253388, 4, by decide, by omega, by omega⟩
      | exact ⟨4611686018427387903, 5, by decide, by omega, by omega⟩
      | exact ⟨13835058055282163711, 5, by decide, by omega, by omega⟩

/-- A vanishing offset is an exact midpoint, and the only one Żmij does not
    special-case is three quarters, whose digit is eight. -/
private theorem digit_mid_even {φ d : ℕ} (hφ : φ < 2 ^ 64) (hd : d ≤ 10)
    (hsp : φ ≠ 2 ^ 62) (hv : (0 : ℤ) = 20 * φ + 2 ^ 64 - 2 * d * 2 ^ 64) :
    d % 2 = 0 := by
  interval_cases d <;> omega

/-- Żmij's digit is a nearest one on the grid at `k`, ties to even. Away from a
    boundary the `+6` cannot matter; at one the fraction is one of eleven
    constants, and all but the quarter are refuted. -/
theorem digit_nearest (f : ℕ) (e : ℤ) (hr : binary64.Regular f e) :
    (-(step e : ℤ) ≤ digitDist f e ∧ digitDist f e ≤ step e)
      ∧ (digitDist f e = step e ∨ digitDist f e = -(step e : ℤ)
          → (toDecimalCandidates f e).digit % 2 = 0) := by
  have hs := (exponent_shift_range e hr.range).2
  have hedge : (0 : ℤ) < edge e := by exact_mod_cast edge_pos e
  have herr : (0 : ℤ) ≤ (err f e : ℤ) := Int.natCast_nonneg _
  have herrlt : 20 * (err f e : ℤ) < 21 * edge e := by
    exact_mod_cast err_lt f e hr
  have hmul : (step e : ℤ) = edge e * 2 ^ 64 := by exact_mod_cast step_eq_edge e
  obtain ⟨φ, hφ⟩ : ∃ φ, fractionalPart f e = φ := ⟨_, rfl⟩
  obtain ⟨d, hd⟩ : ∃ d, (toDecimalCandidates f e).digit = d := ⟨_, rfl⟩
  have hφlt : φ < 2 ^ 64 := by
    rw [← hφ, fractionalPart]
    exact Nat.mod_lt _ (by norm_num)
  -- Twenty times the fraction, less the boundary below it: the whole
  -- argument is a comparison of this offset against the truncation.
  obtain ⟨v, hv⟩ : ∃ v : ℤ, v = 20 * φ + 2 ^ 64 - 2 * d * 2 ^ 64 := ⟨_, rfl⟩
  have hidlo : digitDist f e + step e = edge e * v + 20 * err f e := by
    have hgap : (gap f e : ℤ) = edge e * φ + err f e := by
      rw [← hφ]
      exact_mod_cast gap_eq_edge_fraction f e hs
    rw [digitDist, hd, hgap, hmul, hv]
    push_cast
    ring
  have hidhi : digitDist f e - step e
      = edge e * (v - 2 ^ 65) + 20 * err f e := by
    linear_combination hidlo - 2 * hmul
  -- Name the two sides so that the bounds below are linear in them.
  obtain ⟨A, hA⟩ : ∃ A : ℤ, A = edge e * v + 20 * err f e := ⟨_, rfl⟩
  obtain ⟨B, hB⟩ : ∃ B : ℤ, B = edge e * (v - 2 ^ 65) + 20 * err f e := ⟨_, rfl⟩
  rw [← hA] at hidlo
  rw [← hB] at hidhi
  have hmulpos : (0 : ℤ) < step e := by exact_mod_cast step_pos e
  rw [hd]
  by_cases hsp : φ = 2 ^ 62
  · -- The quarter: the one exact midpoint, whose digit Żmij rounds to even.
    have hd2 : d = 2 := by
      rw [← hd, table_digit_eq f e hr, hφ, roundedDigit]
      simp only [hsp, ↓reduceIte]
    have hv2 : v = 2 ^ 65 := by rw [hv, hsp, hd2]; ring
    have hzero : err f e = 0 := by
      by_contra hne
      exact no_digit_high f e hr (w := 4611686018427387904) (j := 0) (by decide)
        (hφ.trans hsp) (by omega) (by omega)
    have hB0 : B = 0 := by
      rw [hB, hv2, show ((err f e : ℕ) : ℤ) = 0 from by rw [hzero]; rfl]
      ring
    exact ⟨⟨by omega, by omega⟩, fun _ => by omega⟩
  -- Away from it the digit is the biased quotient, which bounds the offset both
  -- ways: the `+6` is worth twelve of the doubled units, so the offset misses
  -- the boundary below by at most that and the one above by at least it.
  have hdq : d = (φ * 10 + biasedHalf) / 2 ^ 64 := by
    rw [← hd, table_digit_eq f e hr, hφ, roundedDigit]
    simp only [hsp, ↓reduceIte]
  rw [biasedHalf] at hdq
  obtain ⟨hd10, hvlo, hvhi⟩ :=
    digit_offset_bounds (m := (φ * 10 + (2 ^ 63 + 6)) % 2 ^ 64) hφlt
      (by rw [hdq]; exact Nat.div_add_mod _ _) (Nat.mod_lt _ (by norm_num)) hv
  clear hdq
  by_cases hneg : v < 0
  · -- Below the boundary.
    obtain ⟨w, j, hmem, hw, hvj⟩ := digit_low_cases hφlt hd10 hv hvlo hneg
    have hfar : 0 < A := by
      by_contra hcon0
      have hcon := not_lt.mp hcon0
      rw [hA, hvj] at hcon
      refine no_digit_low f e hr hmem (by omega) (res_le_of_err f e ?_)
      have h : 20 * (err f e : ℤ) ≤ 4 * (j : ℤ) * edge e := by linarith
      exact_mod_cast h
    have hbelow : B < 0 := by
      have h : edge e * (v - 2 ^ 65) ≤ edge e * (-(2 ^ 65)) :=
        mul_le_mul_of_nonneg_left (by omega) (le_of_lt hedge)
      rw [hB]
      linarith
    exact ⟨⟨by omega, by omega⟩, fun h => by omega⟩
  by_cases hpos : 2 ^ 65 - 20 ≤ v
  · -- Above it.
    obtain ⟨w, j, hmem, hw, hvj⟩ := digit_high_cases hφlt hd10 hv hpos hvhi
    have hj : 4 ≤ j := by omega
    have hfar : B < 0 := by
      by_contra hcon0
      have hcon := not_lt.mp hcon0
      rw [hB, hvj] at hcon
      have hreach : 4 * j * edge e ≤ 20 * err f e := by
        have h : (4 * j * edge e : ℤ) ≤ 20 * (err f e : ℤ) := by linarith
        exact_mod_cast h
      have hjpos : 0 < 4 * j * edge e :=
        Nat.mul_pos (Nat.mul_pos (by norm_num) (by omega)) (edge_pos e)
      exact no_digit_high f e hr hmem (by omega)
        (res_ge_of_err f e hr hreach) (by omega)
    have habove : 0 < A := by
      have h : edge e * (2 ^ 65 - 20) ≤ edge e * v :=
        mul_le_mul_of_nonneg_left hpos (le_of_lt hedge)
      rw [hA]
      linarith
    exact ⟨⟨by omega, by omega⟩, fun h => by omega⟩
  -- Everywhere else the offset clears the truncation on both sides. The one
  -- exception is a vanishing offset, which is an exact midpoint: there the
  -- fraction is three quarters and the digit is eight.
  have hedgev : 0 ≤ edge e * v := mul_nonneg (le_of_lt hedge) (by omega)
  have hbelow : B < 0 := by
    have h : edge e * (v - 2 ^ 65) ≤ edge e * (-21) :=
      mul_le_mul_of_nonneg_left (by omega) (le_of_lt hedge)
    rw [hB]
    linarith
  have habove : 0 ≤ A := by rw [hA]; omega
  refine ⟨⟨by omega, by omega⟩, fun h => ?_⟩
  -- A midpoint forces the offset to vanish, and only one fraction does.
  have hv0 : v = 0 := by
    have hz : edge e * v = 0 := by rw [hA] at habove hidlo; omega
    rcases mul_eq_zero.mp hz with h' | h'
    · omega
    · exact h'
  exact digit_mid_even hφlt hd10 hsp (by rw [← hv0]; exact hv)

/-! ## The grid at k

Everything so far has been stated on the grid at `k+1`, where the coarse
candidates live, or in integers. The exact method asks for the grid at `k`, one
power of ten finer: `10^(-k) = 10·10^(-k-1)`. Two things have to be said there.
One ULP has to fall between one and ten steps of it, which is what makes `k` the
right exponent to report and is checked per exponent. And Żmij's output has to
be read on it: the coarse candidates as multiples of ten, and the fine one as
the integral part with the digit appended.
-/

/-- The power of ten is normalized against the grid at `k`: in integers, one
    ULP is `10·num` where one step of that grid is worth `step`, so a ULP spans
    at least one step and less than ten. -/
def gridHolds (e : ℤ) : Bool :=
  decide (num e < step e ∧ step e ≤ 10 * num e)

theorem grid_all : ∀ e ∈ Finset.Icc (-1074 : ℤ) 971, gridHolds e = true := by
  decide +kernel

theorem grid_bounds (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    num e < step e ∧ step e ≤ 10 * num e := by
  have hcert := grid_all e (by simpa [Finset.mem_Icc] using he)
  simpa only [gridHolds, decide_eq_true_eq] using hcert

/-- The grid at `k` is the one at `k+1` scaled by ten. -/
private theorem grid_pow (e : ℤ) :
    (10 : ℚ) ^ (-binary64.decimalExponent e)
      = 10 * 10 ^ (-(binary64.decimalExponent e + 1)) := by
  rw [show -binary64.decimalExponent e
        = -(binary64.decimalExponent e + 1) + 1 from by ring,
    zpow_add_one₀ (by norm_num : (10 : ℚ) ≠ 0)]
  ring

/-- The value on the grid at `k`, multiplied through by `step`. -/
theorem value_scaled_grid (f : ℕ) (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    value f e * 10 ^ (-binary64.decimalExponent e) * (step e : ℚ)
      = ((10 * (f * num e) : ℕ) : ℚ) := by
  have h := value_scaled f e he
  rw [grid_pow]
  push_cast at h ⊢
  linear_combination 10 * h

/-- Half a ULP on the grid at `k`, in integers: five `num`. -/
theorem half_ulp_grid (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    ulp e * 10 ^ (-binary64.decimalExponent e) / 2 * (step e : ℚ)
      = ((5 * num e : ℕ) : ℚ) := by
  have h := half_ulp_scaled e he
  rw [grid_pow]
  push_cast at h ⊢
  linear_combination 5 * h

/-- One ULP is at least one step of the grid at `k` and less than ten. This is
    the one obligation of the exact method that is about `k` alone. -/
theorem ulp_scaled_bounds (e : ℤ) (he : -1074 ≤ e ∧ e ≤ 971) :
    1 ≤ ulp e * 10 ^ (-binary64.decimalExponent e)
      ∧ ulp e * 10 ^ (-binary64.decimalExponent e) < 10 := by
  obtain ⟨hnum, hstep⟩ := grid_bounds e he
  refine ulp_steps_of_int_eq (u' := 10 * num e) (step_pos e) ?_ hstep (by omega)
  have h := half_ulp_grid e he
  push_cast at h ⊢
  linear_combination 2 * h

/-- Żmij's significand: the integral part on the grid at `k`, with the digit
    appended unless a coarse candidate was taken. -/
theorem to_decimal_fst (f : ℕ) (e : ℤ) :
    (toDecimal f e).1
      = (integralPart f e
            + (if (toDecimalCandidates f e).roundUp then 1 else 0)) * 10
        + (if (toDecimalCandidates f e).roundUp
              || (toDecimalCandidates f e).roundDown
            then 0 else (toDecimalCandidates f e).digit) := rfl

/-- A coarse candidate seen on the grid at `k`: ten times its significand one
    grid up. -/
private theorem roundtrips_ten (f : ℕ) (e : ℤ) (c d : ℕ) (h : d = c * 10) :
    Roundtrips f e (d * 10 ^ binary64.decimalExponent e)
      ↔ Roundtrips f e (c * 10 ^ (binary64.decimalExponent e + 1)) := by
  rw [show (d : ℚ) * 10 ^ binary64.decimalExponent e
      = (c : ℚ) * 10 ^ (binary64.decimalExponent e + 1) from by
    rw [zpow_add_one₀ (by norm_num : (10 : ℚ) ≠ 0), h]
    push_cast
    ring]

/-! ## Żmij trims whenever it can

The exact method takes the coarse case exactly when the rounding interval
contains a multiple of ten, that is, when a digit can be dropped. Żmij makes
the same choice through `roundDown` and `roundUp`: it trims when either of its
two coarse candidates round-trips, which is what `round_down_iff_roundtrips`
and `round_up_iff_roundtrips` already say. All that is left is to rule out any
other multiple of ten, and the rounding interval is too narrow to hold one.
-/

/-- A multiple of ten that round-trips is one of Żmij's two coarse candidates.
    Multiplied through by `step`, the lower one sits `10·gap` below the scaled
    value, and that gap is under one step of the coarse grid, let alone one step
    plus half a ULP. This bracket is all `coarse_roundtrip_adjacent` needs. -/
theorem coarse_candidate_cases (f : ℕ) (e : ℤ)
    (hr : binary64.Regular f e) (d : ℕ)
    (h10 : d % 10 = 0)
    (hround : Roundtrips f e (d * 10 ^ binary64.decimalExponent e)) :
    d = integralPart f e * 10 ∨ d = integralPart f e * 10 + 10 := by
  have hmul : (0 : ℚ) < (step e : ℚ) := by exact_mod_cast step_pos e
  have hval := value_scaled_grid f e hr.range
  have hhalf := half_ulp_grid e hr.range
  have hid : (integralPart f e : ℚ) * (step e : ℚ) + (gap f e : ℚ)
      = (f : ℚ) * (num e : ℚ) := by
    exact_mod_cast integral_add_gap f e (exponent_shift_range e hr.range).2
  have hgap0 : (0 : ℚ) ≤ (gap f e : ℚ) := by positivity
  have hnum0 : (0 : ℚ) < (num e : ℚ) := by exact_mod_cast num_pos e hr.range
  have hgaple : (gap f e : ℚ) ≤ (step e : ℚ) := by
    exact_mod_cast gap_le_step f e hr
  push_cast at hval hhalf
  exact coarse_roundtrip_adjacent f e (binary64.decimalExponent e)
    (ulp_scaled_bounds e hr.range).2 (by omega) h10
    (le_of_mul_le_mul_right (by push_cast; linarith) hmul)
    (lt_of_mul_lt_mul_right (by push_cast; linarith) hmul.le) hround

/-- If the rounding interval contains a multiple of ten, Żmij trims. -/
theorem trim_of_coarse_roundtrip (f : ℕ) (e : ℤ)
    (hr : binary64.Regular f e) (d : ℕ)
    (h10 : d % 10 = 0)
    (hround : Roundtrips f e (d * 10 ^ binary64.decimalExponent e)) :
    ((toDecimalCandidates f e).roundUp
      || (toDecimalCandidates f e).roundDown) = true := by
  rw [Bool.or_eq_true]
  rcases coarse_candidate_cases f e hr d h10 hround with rfl | rfl
  · exact Or.inr ((round_down_iff_roundtrips f e hr).mpr
      ((roundtrips_ten f e (integralPart f e) _ rfl).mp hround))
  · exact Or.inl ((round_up_iff_roundtrips f e hr).mpr
      ((roundtrips_ten f e (integralPart f e + 1) _ (by ring)).mp hround))

/-- Trimmed output is a multiple of ten that round-trips: it is whichever of the
    two coarse candidates fired. -/
theorem coarse_output_roundtrips (f : ℕ) (e : ℤ) (hr : binary64.Regular f e)
    (htrim : ((toDecimalCandidates f e).roundUp
      || (toDecimalCandidates f e).roundDown) = true) :
    (toDecimal f e).1 % 10 = 0
      ∧ Roundtrips f e
          ((toDecimal f e).1 * 10 ^ binary64.decimalExponent e) := by
  have hd : (toDecimal f e).1
      = (integralPart f e
          + (if (toDecimalCandidates f e).roundUp then 1 else 0)) * 10 := by
    rw [to_decimal_fst, htrim]
    simp
  rw [hd]
  refine ⟨by omega, ?_⟩
  cases hu : (toDecimalCandidates f e).roundUp
  · -- Only `roundDown` fired, so the integral part itself round-trips.
    have hdown : (toDecimalCandidates f e).roundDown = true := by
      rw [hu, Bool.false_or] at htrim
      exact htrim
    exact (roundtrips_ten f e (integralPart f e) _ (by simp)).mpr
      ((round_down_iff_roundtrips f e hr).mp hdown)
  · exact (roundtrips_ten f e (integralPart f e + 1) _ (by simp)).mpr
      ((round_up_iff_roundtrips f e hr).mp hu)

/-- Untrimmed output is a nearest candidate on the grid at `k`, ties to even:
    the integral part accounts for ten times its own distance and the digit for
    the rest, which is `digit_nearest`. -/
theorem fine_output_nearest (f : ℕ) (e : ℤ) (hr : binary64.Regular f e)
    (htrim : ((toDecimalCandidates f e).roundUp
      || (toDecimalCandidates f e).roundDown) = false) :
    let x := value f e * 10 ^ (-binary64.decimalExponent e)
    |((toDecimal f e).1 : ℚ) - x| ≤ 1 / 2
      ∧ (|((toDecimal f e).1 : ℚ) - x| = 1 / 2 → (toDecimal f e).1 % 2 = 0) := by
  intro x
  have hu : (toDecimalCandidates f e).roundUp = false :=
    (Bool.or_eq_false_iff.mp htrim).1
  have hd : (toDecimal f e).1
      = integralPart f e * 10 + (toDecimalCandidates f e).digit := by
    rw [to_decimal_fst, htrim, hu]
    simp
  have hid : (integralPart f e : ℤ) * step e + gap f e = f * num e := by
    exact_mod_cast integral_add_gap f e (exponent_shift_range e hr.range).2
  obtain ⟨hle, -, heq⟩ := scaled_cmp_of_int_eq (c := (toDecimal f e).1)
    (scale := step e) (a := 2) (b := step e) (x := x) (thr := 1 / 2)
    (dist := 10 * (gap f e : ℤ) - (toDecimalCandidates f e).digit * step e)
    (step_pos e) two_pos (value_scaled_grid f e hr.range) (by push_cast; ring)
    (by rw [hd]; push_cast; linear_combination 10 * hid)
  have hdd : 2 * (10 * (gap f e : ℤ)
      - (toDecimalCandidates f e).digit * step e) = digitDist f e := by
    rw [digitDist]
    ring
  push_cast at hle heq
  rw [hdd] at hle heq
  obtain ⟨⟨hlo, hhi⟩, heven⟩ := digit_nearest f e hr
  refine ⟨hle.mpr ⟨hlo, hhi⟩, fun h => ?_⟩
  have hdig := heven (heq.mp h)
  omega

/-! ## Żmij refines the exact method

Nothing above is needed beyond `ulp_scaled_bounds` and the three semantic
obligations `coarse_output_roundtrips`, `fine_output_nearest`, and
`trim_of_coarse_roundtrip`. In particular no claim is made that Żmij's packed
decisions agree with the exact ones: its truncated comparisons are matched to
the existence of an exact coarse candidate, not to any exact comparison, and
its one special case is a packed midpoint rather than an exact one.
-/

/-- Żmij implements the exact method: it trims exactly when an exact coarse
    candidate exists. One direction is `trim_of_coarse_roundtrip`; the other
    holds because trimmed output is itself a multiple of ten that
    round-trips. -/
theorem exact_candidate (f : ℕ) (e : ℤ) (hr : binary64.Regular f e) :
    let (d, k) := toDecimal f e
    ExactCandidate f e k d := by
  show ExactCandidate f e (binary64.decimalExponent e) (toDecimal f e).1
  by_cases htrim : ((toDecimalCandidates f e).roundUp
      || (toDecimalCandidates f e).roundDown) = true
  · exact Or.inl (coarse_output_roundtrips f e hr htrim)
  · rw [Bool.not_eq_true] at htrim
    obtain ⟨hle, heven⟩ := fine_output_nearest f e hr htrim
    refine Or.inr ⟨fun ⟨d, h10, hround⟩ => ?_, hle, heven⟩
    rw [trim_of_coarse_roundtrip f e hr d h10 hround] at htrim
    exact Bool.noConfusion htrim

/-- Żmij is correct on regularly spaced positive binary64 values: after removing
    trailing zeros its output is a shortest decimal representation that
    round-trips, and it is correctly rounded on its own decimal grid. -/
theorem correct (f : ℕ) (e : ℤ) (hr : binary64.Regular f e) :
    let (d, k) := toDecimal f e
    let (d', k') := reduceDecimal d k
    Shortest f e d' k' ∧ CorrectlyRounded f e d' k' := by
  obtain ⟨hfine, hcoarse⟩ := ulp_scaled_bounds e hr.range
  exact exact_candidate_correct f e (binary64.decimalExponent e) hr.pos
    hfine hcoarse (exact_candidate f e hr)

end Zmij
