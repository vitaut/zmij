import Mathlib

-- Exact rational value represented by binary significand f and exponent e.
def value (f : ℕ) (e : ℤ) : ℚ := f * 2 ^ e

-- Lower rounding boundary for a regularly spaced floating-point value (m⁻).
def lower (f : ℕ) (e : ℤ) : ℚ := (f - 1 / 2) * 2 ^ e

-- Upper rounding boundary (m⁺).
def upper (f : ℕ) (e : ℤ) : ℚ := (f + 1 / 2) * 2 ^ e

-- Whether the exact rational result r rounds to the regularly spaced value
-- f · 2^e under round-to-nearest, ties-to-even.
def Roundtrips (f : ℕ) (e : ℤ) (r : ℚ) : Prop :=
  if f % 2 = 0 then
    lower f e ≤ r ∧ r ≤ upper f e
  else
    lower f e < r ∧ r < upper f e

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

structure DecimalCandidates where
  k : ℤ
  decOne : ℕ
  decTen : ℕ
  roundD0 : Bool
  roundU0 : Bool

def toDecimalCandidates (f : ℕ) (e : ℤ) : DecimalCandidates :=
  let k := e * 315653 / 2 ^ 20
  let h := Int.toNat (e + (-k * 217707) / 2 ^ 16)

  let p10 := power10Significand (-k)
  let p10Hi := p10 / 2 ^ 64

  let cb := f * 2 ^ (h + 1)
  let product := cb * p10 / 2 ^ 64
  let sigHi := product / 2 ^ 64
  let sigLo := product % 2 ^ 64

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

-- The cached power-of-10 significand is the truncation of the exact scaled value.
theorem power10_significand_bounds (k : ℤ) :
    let x := (10 : ℚ) ^ k * 2 ^ (128 - power10Exponent k)
    (power10Significand k : ℚ) ≤ x ∧ x < power10Significand k + 1 := by
  dsimp [power10Significand]
  exact ⟨Nat.floor_le (by positivity), Nat.lt_floor_add_one _⟩

-- The shift used by yy's regular path is less than 4.
theorem decimal_shift_lt_four
    (f : ℕ) (e : ℤ)
    (h : Regular f e) :
    let k := e * 315653 / 2 ^ 20
    Int.toNat (e + (-k * 217707) / 2 ^ 16) < 4 := by
  dsimp
  rcases h with ⟨_, _, hlo, hhi⟩
  omega

-- The decimal significand produced by yy is within half a scaled ULP
-- of the exact value, with ties allowed only for even significands.
theorem decimal_significand_distance
    (f : ℕ) (e : ℤ)
    (h : Regular f e) :
    let (d, k) := toDecimal f e
    let x := value f e * 10 ^ (-k)
    let u := (2 : ℚ) ^ e * 10 ^ (-k)
    if f % 2 = 0 then
      |d - x| ≤ u / 2
    else
      |d - x| < u / 2 := by
  let c := toDecimalCandidates f e

  have hto :
      toDecimal f e =
        (if c.roundD0 || c.roundU0 then c.decTen else c.decOne, c.k) := rfl
  rw [hto]

  -- Match the goal's exact inverse notation to prevent type mismatch
  let x := value f e * (10 ^ c.k)⁻¹
  let u := (2 : ℚ) ^ e * (10 ^ c.k)⁻¹

  let InRange (dec : ℕ) : Prop :=
    if f % 2 = 0 then |dec - x| ≤ u / 2 else |dec - x| < u / 2

  have hone : InRange c.decOne := sorry
  have hten_d0 (hd0 : c.roundD0 = true) : InRange c.decTen := sorry
  have hten_u0 (hu0 : c.roundU0 = true) : InRange c.decTen := sorry

  by_cases hd0 : c.roundD0 = true
  · simp [hd0]
    exact hten_d0 hd0
  · have hd0' : c.roundD0 = false := by revert hd0; cases c.roundD0 <;> simp
    by_cases hu0 : c.roundU0 = true
    · simp [hd0', hu0]
      exact hten_u0 hu0
    · have hu0' : c.roundU0 = false := by revert hu0; cases c.roundU0 <;> simp
      simp [hd0', hu0']
      exact hone

-- The decimal significand produced by yy lies within the rounding interval
-- after scaling by the decimal exponent.
theorem decimal_significand_in_interval
    (f : ℕ) (e : ℤ)
    (h : Regular f e) :
    let (d, k) := toDecimal f e
    let p10 := 10 ^ (-k)
    if f % 2 = 0 then
      lower f e * p10 ≤ d ∧ d ≤ upper f e * p10
    else
      lower f e * p10 < d ∧ d < upper f e * p10 := by
  rcases hdk : toDecimal f e with ⟨d, k⟩
  dsimp only

  let x := value f e * 10 ^ (-k)
  let u := (2 : ℚ) ^ e * 10 ^ (-k)

  have hd := decimal_significand_distance f e h

  have hlower : lower f e * 10 ^ (-k) = x - u / 2 := by
    dsimp [x, u, lower, value]; ring

  have hupper : upper f e * 10 ^ (-k) = x + u / 2 := by
    dsimp [x, u, upper, value]; ring

  split_ifs with heven <;> rw [hlower, hupper]
  · have hd' : |d - x| ≤ u / 2 := by simpa [hdk, x, u, heven] using hd
    obtain ⟨hl, hr⟩ := abs_le.mp hd'
    constructor <;> linarith
  · have hd' : |d - x| < u / 2 := by simpa [hdk, x, u, heven] using hd
    obtain ⟨hl, hr⟩ := abs_lt.mp hd'
    constructor <;> linarith

-- The decimal representation produced by yy round-trips to the original value.
theorem yy_roundtrips
    (f : ℕ) (e : ℤ)
    (h : Regular f e) :
    let (d, k) := toDecimal f e
    Roundtrips f e (d * 10 ^ k) := by
  rcases hdk : toDecimal f e with ⟨d, k⟩
  have hi := decimal_significand_in_interval f e h
  simp only [hdk] at hi

  have hpow_ne : (10 : ℚ) ^ k ≠ 0 := by positivity
  have hcancel : (10 : ℚ) ^ (-k) * 10 ^ k = 1 := by
    simpa only [zpow_neg] using inv_mul_cancel₀ hpow_ne

  have hrescale (x : ℚ) : x = x * 10 ^ (-k) * 10 ^ k := by
    rw [mul_assoc, hcancel, mul_one]

  simp only [Roundtrips]

  -- Split parity and prove both bounds from the scaled interval.
  split_ifs at hi ⊢ <;> constructor <;>
    first
    | rw [hrescale (lower f e)]
      gcongr
      exact hi.1
    | rw [hrescale (upper f e)]
      gcongr
      exact hi.2
