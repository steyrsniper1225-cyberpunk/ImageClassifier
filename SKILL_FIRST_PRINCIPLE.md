---
name: first-principles
description: This skill should be used when the user asks to "analyze from first principles", "think from scratch", "question this design", "is this the right approach", "why are we doing it this way", "is there a better solution", "challenge assumptions", or in Korean "1원칙으로 분석", "제1원리", "근본부터 다시", "원점에서 다시 생각", "이 설계 합리적인가", "이 방향이 맞나", "왜 이렇게 하는 건가", "더 나은 방법 없나", "가정을 의심", or needs to evaluate architectural decisions, design choices, or problem-solving approaches without relying on analogies or conventions.
version: 0.3.0
---

# First Principles Thinking

A systematic approach to decomposing complex problems into fundamental truths and reasoning up from there, avoiding the trap of reasoning by analogy.

## When to Use This Skill

- Evaluating whether an architecture or design is truly optimal
- Questioning "best practices" that may not fit the current context
- Breaking through when conventional solutions feel inadequate
- Making foundational decisions that will have long-term impact
- Challenging inherited assumptions in legacy systems
- Designing new systems without cargo-culting existing patterns

## Core Process

### Phase 1: Identify the Problem's Essence

Strip away implementation details to find the core problem:

1. **State the problem clearly** - What exactly needs to be solved?
2. **Separate symptoms from causes** - Is this the real problem or a manifestation?
3. **Define success criteria** - What would a perfect solution achieve?

**Key Questions:**
- What is the fundamental job to be done here?
- If this system didn't exist, what would users actually need?
- What outcome matters, independent of how we get there?

### Phase 2: Challenge All Assumptions

Identify and question every assumption:

1. **List explicit assumptions** - What are we taking as given?
2. **Surface implicit assumptions** - What conventions are we following without questioning?
3. **Test each assumption** - Is this actually a constraint, or just how it's always been done?

**Assumption Categories:**

| Category | Question to Ask |
|----------|----------------|
| Technical | "Must we use this technology/pattern?" |
| Business | "Is this requirement actually fixed?" |
| Resource | "Are these constraints real or perceived?" |
| Historical | "Why was this decision made originally?" |

**Red Flags (likely false assumptions):**
- "We've always done it this way"
- "Industry standard says..."
- "Everyone uses X for this"
- "That's too simple to work"

### Phase 3: Establish Ground Truths

Identify the irreducible facts:

1. **Physics/Math constraints** - What cannot be violated?
2. **Business invariants** - What must remain true for the business?
3. **User needs** - What does the user fundamentally require?

**Ground Truth Test:**
- Can this be further decomposed?
- Is this provably true, not just commonly believed?
- Would violating this definitely cause failure?

### Phase 4: Reason Upward

Build solutions from ground truths:

1. **Start minimal** - What's the simplest thing that satisfies ground truths?
2. **Add only what's necessary** - Each addition must justify itself
3. **Challenge each layer** - Does this layer earn its complexity?

**Building Blocks Approach:**
```
Ground Truth → Minimal Solution → Justified Additions → Final Design
     ↑              ↑                    ↑
  (proven)     (sufficient)        (each defended)
```

### Phase 5: Validate the Reasoning

Ensure the solution is sound:

1. **Trace back to ground truths** - Can every design decision be traced to a fundamental need?
2. **Identify weak links** - Where does the reasoning rely on assumptions?
3. **Stress test** - What would break this solution?

## Output Format

When applying first principles thinking, structure the analysis as:

```markdown
## First Principles Analysis: [Topic]

### 1. Problem Essence
**Core problem:** [One sentence]
**Success criteria:** [Measurable outcomes]

### 2. Assumptions Challenged
| Assumption | Challenge | Evidence / How to Verify | Verdict |
|------------|-----------|---------------------------|---------|
| [Assumption] | [Why question it] | [What was actually observed: code, logs, docs, numbers] | [Keep/Discard/Modify/Unverified] |

> A Keep/Discard/Modify verdict requires a non-empty Evidence cell. If nothing was actually observed, the verdict stays **Unverified**.

### 3. Ground Truths
- [Irreducible fact 1]
- [Irreducible fact 2]
- [Irreducible fact 3]

### 4. Reasoning Chain
Ground Truth → [Step 1] → [Step 2] → Solution

### 5. Conclusion
**Recommended approach:** [Description]
**Key insight:** [What the first principles analysis revealed]
**Trade-offs acknowledged:** [What we're accepting]
```

### Language

Write the analysis in the language the user is using. For Korean users, use the template below (한국어 존댓말):

```markdown
## 1원칙 분석: [주제]

### 1. 문제의 본질
**핵심 문제:** [한 문장]
**성공 기준:** [측정 가능한 결과]

### 2. 도전한 가정
| 가정 | 왜 의심하는가 | 근거/확인방법 | 판정 |
|------|--------------|--------------|------|
| [가정] | [의심 근거] | [실제로 확인한 것: 코드, 로그, 문서, 수치] | [유지/폐기/수정/확인 필요] |

> 유지, 폐기, 수정 판정은 근거/확인방법 칸이 채워져 있을 때만 내립니다. 실제로 관측한 게 없으면 판정은 **확인 필요**로 남깁니다.

### 3. 근본 사실
- [더 쪼갤 수 없는 사실 1]
- [더 쪼갤 수 없는 사실 2]
- [더 쪼갤 수 없는 사실 3]

### 4. 추론 사슬
근본 사실 → [1단계] → [2단계] → 해법

### 5. 결론
**권고안:** [설명]
**핵심 통찰:** [1원칙 분석으로 드러난 것]
**감수하는 트레이드오프:** [수용하는 비용]
```

## Common Patterns

### Pattern: The Complexity Trap

**Symptom:** Solution is more complex than the problem warrants

**First Principles Check:**
1. Remove one component - does the system still solve the core problem?
2. If yes, that component wasn't essential
3. Repeat until removal breaks core functionality

### Pattern: The Analogy Trap

**Symptom:** "Company X does it this way, so we should too"

**First Principles Check:**
1. What problem was Company X actually solving?
2. Is our problem identical in all relevant dimensions?
3. What constraints did they have that we don't (and vice versa)?

### Pattern: The Legacy Trap

**Symptom:** Maintaining compatibility with decisions that no longer serve us

**First Principles Check:**
1. What was the original reason for this decision?
2. Do those conditions still exist?
3. What's the true cost of change vs. cost of maintaining?

## Integration with Other Thinking Tools

First principles thinking works well with these complementary approaches:

| Tool | When to Combine | How |
|------|-----------------|-----|
| **Trade-off Analysis** | After identifying ground truths | Evaluate options against fundamentals |
| **5-Whys** | When assumptions surface | Dig deeper to find root causes |
| **Pre-mortem** | Before finalizing solution | Stress-test reasoning chain |
| **Hypothesis Testing** | When ground truths are uncertain | Validate assumptions empirically |

## Boundaries

**Will:**
- Challenge assumptions systematically
- Identify ground truths from first principles
- Build reasoning chains from fundamentals
- Reveal when conventional wisdom doesn't apply

**Will Not:**
- Dismiss all existing solutions as wrong
- Spend unlimited time on every decision (reserve for important choices)
- Ignore practical constraints in favor of theoretical purity
- Guarantee the "best" solution (reveals better reasoning, not perfect answers)

## Quick Reference

**The First Principles Checklist:**
- [ ] Problem stated in terms of outcomes, not solutions
- [ ] All assumptions explicitly listed
- [ ] Each assumption challenged and justified
- [ ] Ground truths identified and verified
- [ ] Solution built up from ground truths only
- [ ] Every design decision traceable to a ground truth
- [ ] Reasoning chain documented

## Additional Resources

### Reference Files
- **`references/elon-musk-examples.md`** - Real-world examples from SpaceX and Tesla
- **`references/software-examples.md`** - Software engineering applications

### Example Files
- **`examples/architecture-review.md`** - First principles analysis of a microservices decision
