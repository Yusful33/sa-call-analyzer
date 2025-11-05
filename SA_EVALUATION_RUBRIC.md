# 🎯 SA Performance Evaluation Rubric

## Overview

Your system evaluates Solution Architects using **3 specialized AI agents** that analyze different aspects of call performance, then a 4th agent compiles everything into actionable feedback.

---

## 📋 Evaluation Framework

### Core Philosophy

The evaluation is based on the **Command of the Message** sales methodology combined with technical excellence and discovery skills.

**Key Principle**: Small talk and rapport building (1-3 minutes) is **ENCOURAGED and VALUABLE** - not penalized.

---

## 🤖 The 4-Agent Evaluation System

### **Agent 1: SA Identifier** 🔍
**Purpose**: Determine who the Solution Architect is in the call

**How it works**:
- Analyzes conversation patterns
- Looks for technical discussions, architecture talks, integration details
- Identifies the person doing technical scoping and problem-solving

**Output**: SA's name + confidence level (high/medium/low)

---

### **Agent 2: Senior Technical Architect & Evaluator** 🛠️
**Purpose**: Assess technical performance

**What it evaluates**:

| Dimension | What "Good" Looks Like | What "Bad" Looks Like |
|-----------|------------------------|----------------------|
| **Technical Depth** | - Explains complex concepts clearly<br>- Provides accurate technical details<br>- Demonstrates deep product knowledge | - Vague or superficial explanations<br>- Technical inaccuracies<br>- Can't answer technical questions |
| **Architecture Discussion** | - Discusses integration points<br>- Considers scalability and architecture<br>- Addresses technical constraints | - Ignores architecture implications<br>- No discussion of how it fits into their stack<br>- Doesn't explore technical requirements |
| **Demo Quality** | - Well-prepared and relevant demo<br>- Tied to customer's specific use case<br>- Anticipates questions | - Generic demo not tailored to customer<br>- Technical issues during demo<br>- Doesn't showcase relevant features |

**Example Good Performance**:
> [12:45] "For your use case with high-volume data ingestion, I'd recommend configuring the batch processor with a 10-second window. Given your Kafka setup, we can leverage the native connector I'm showing here..."

**Example Poor Performance**:
> [12:45] "Yeah, we can handle data ingestion. It's pretty straightforward."

---

### **Agent 3: Sales Methodology & Discovery Expert** 💡
**Purpose**: Evaluate discovery skills and Command of the Message framework execution

#### **Part A: Discovery & Engagement**

**What it evaluates**:

| Dimension | What "Good" Looks Like | What "Bad" Looks Like |
|-----------|------------------------|----------------------|
| **Discovery Question Quality** | - Open-ended questions<br>- Probing follow-ups<br>- Gets beneath surface answers<br>- Uncovers root causes | - Only yes/no questions<br>- No follow-up questions<br>- Superficial questioning<br>- Misses opportunities to dig deeper |
| **Active Listening** | - Acknowledges customer responses<br>- References previous comments<br>- Builds on what customer said<br>- Paraphrases to confirm understanding | - Interrupts customer<br>- Doesn't acknowledge responses<br>- Moves on too quickly<br>- Doesn't confirm understanding |
| **Engagement** | - Conversational and natural<br>- Responds to customer cues<br>- Adapts based on customer interest<br>- Creates dialogue, not monologue | - Robotic or scripted<br>- Ignores customer signals<br>- One-way presentation<br>- Doesn't adapt to customer |
| **Rapport Building** | - Appropriate small talk (1-3 min OK!)<br>- Finds common ground<br>- Warm and personable<br>- Builds trust | - Too formal or cold<br>- Rushes into business<br>- No personal connection<br>- Feels transactional |

**Example Good Discovery**:
> [5:23] SA: "You mentioned your team struggles with data quality. Can you walk me through a recent example where bad data caused a problem?"
>
> [6:10] SA: "So if I'm understanding correctly, the sales team couldn't trust the pipeline numbers, which led to missed forecasts. What was the downstream impact on the business?"

**Example Poor Discovery**:
> [5:23] SA: "Do you have data quality issues?"
>
> [5:25] SA: "Okay, cool. Let me show you our data validation features..."

---

#### **Part B: Command of the Message Framework**

This is a **proven B2B sales methodology** that focuses on four key areas:

### 🎯 **1. Problem Identification**

**What it means**: Uncovering real, painful business problems (not just feature requests)

**Good Example**:
> [8:15] SA: "So when your observability pipeline goes down, what happens to your incident response times? ... So you're essentially flying blind for 20-30 minutes while engineers scramble to figure out what's wrong. And that's costing you customer trust and potential revenue."

**Poor Example**:
> [8:15] SA: "Do you need observability?" → Doesn't dig into WHY or what problem it solves

**Questions to uncover problems**:
- "What happens when X fails?"
- "What's the business impact of Y?"
- "How much time/money does Z cost you?"
- "What would happen if you could eliminate this problem?"

---

### 🏆 **2. Differentiation**

**What it means**: Articulating unique value vs competitors (not just features)

**Good Example**:
> [15:30] SA: "Unlike traditional APM tools that only give you pre-aggregated metrics, we capture raw trace data which means you can ask questions you didn't know you'd need to ask 6 months from now. That's crucial when you're debugging production issues that didn't exist when you set up your dashboards."

**Poor Example**:
> [15:30] SA: "We have really good dashboards and our UI is easy to use." → Generic claims that any competitor could make

**How to differentiate well**:
- Explain *why* your approach is different (not just *what* is different)
- Tie differentiation to the customer's specific problem
- Use proof points (customer stories, data)
- Contrast with alternatives (without bashing competitors)

---

### 📊 **3. Proof & Evidence**

**What it means**: Providing concrete evidence that your solution works (not just claims)

**Types of proof**:
- **Customer stories**: "Company X had the same challenge and reduced MTTR by 40%"
- **Metrics**: "Customers typically see 3x improvement in query speed"
- **Demos**: Live demonstration tied to their specific use case
- **Data**: "In our benchmark tests, we processed 1M events/sec"
- **Case studies**: "Let me share how Acme Corp solved this exact problem"

**Good Example**:
> [22:18] SA: "Let me show you how Datadog uses our platform. They had a similar architecture - microservices on Kubernetes with intermittent latency spikes. After implementing our solution, they reduced their P95 latency from 800ms to 200ms within 2 weeks. Here's the actual dashboard they shared..."

**Poor Example**:
> [22:18] SA: "Yeah, lots of companies use us. It works really well." → No specifics, no proof

---

### 🔧 **4. Required Capabilities**

**What it means**: Connecting technical features to business outcomes

**The Bridge**: `Technical Capability → Enables → Business Value`

**Good Example**:
> [28:44] SA: "Our distributed tracing capability [TECHNICAL] means you can trace a request across all 47 of your microservices [WHAT IT DOES], which allows your engineers to identify the root cause service in under 2 minutes instead of 45 minutes [BUSINESS VALUE]. That translates to faster incident resolution, less downtime, and fewer engineers pulled into war rooms [BUSINESS IMPACT]."

**Poor Example**:
> [28:44] SA: "We have distributed tracing." → Just the feature, no connection to value

**Formula for Required Capabilities**:
1. **Feature**: "We have X capability"
2. **Function**: "Which allows you to Y"
3. **Value**: "So you can achieve Z business outcome"

---

### **Agent 4: Executive Performance Coach** 📝
**Purpose**: Synthesize everything into actionable feedback

**What it produces**:

1. **Top 3-5 Actionable Insights**
   - **Category**: Which skill area (Discovery, Technical, Differentiation, etc.)
   - **Severity**: Critical, Important, or Minor
   - **Timestamp**: Exact moment in the call
   - **What Happened**: Specific behavior observed
   - **Why It Matters**: Business/performance impact
   - **Better Approach**: Alternative way to handle it
   - **Example Phrasing**: Exact words to use next time

2. **Strengths**: What the SA did well (to reinforce)

3. **Improvement Areas**: Focus areas for development

4. **Key Moments**: Significant moments from the call with timestamps

---

## 🎓 Severity Levels Explained

### **Critical** 🔴
- **Impact**: Major missed opportunity or serious mistake
- **Examples**:
  - Failed to identify a major business problem customer mentioned
  - Gave inaccurate technical information
  - Made promises the product can't deliver

### **Important** 🟡
- **Impact**: Moderate missed opportunity affecting call effectiveness
- **Examples**:
  - Surface-level discovery (didn't dig deeper)
  - No differentiation from competitors
  - Generic demo not tailored to customer

### **Minor** 🔵
- **Impact**: Small refinement that would improve performance
- **Examples**:
  - Could have used better phrasing
  - Missed a follow-up question
  - Could have provided more concrete metrics

---

## 📊 How Insights Are Categorized

| Category | What It Covers |
|----------|---------------|
| **Discovery Depth** | Quality of questions and uncovering needs |
| **Problem Identification** | Finding real business problems |
| **Technical Communication** | Explaining technical concepts clearly |
| **Differentiation** | Articulating unique value |
| **Proof & Evidence** | Providing concrete evidence |
| **Value Articulation** | Connecting features to business outcomes |
| **Active Listening** | Responding to customer cues |
| **Demo Quality** | Effectiveness of demonstrations |

---

## 💡 What Makes a "Good" vs "Bad" SA Call

### ✅ Characteristics of an **Excellent** Call

1. **Discovery First**: Asks thoughtful questions before pitching
2. **Business-Focused**: Talks about business problems, not just features
3. **Evidence-Based**: Backs up claims with proof points
4. **Customer-Centric**: Adapts demo/conversation to customer's needs
5. **Clear Value**: Customer understands WHY they should care
6. **Technical Depth**: Demonstrates expertise when needed
7. **Differentiated**: Customer understands what makes you unique
8. **Actionable**: Customer knows what next steps are

### ❌ Characteristics of a **Poor** Call

1. **Feature Dumping**: Lists features without connecting to value
2. **Surface-Level**: Doesn't dig into real problems
3. **Generic**: Could be any competitor's pitch
4. **No Proof**: Makes claims without evidence
5. **Technical Jargon**: Confuses customer with complexity
6. **One-Way**: Monologue instead of dialogue
7. **Misses Cues**: Doesn't respond to customer signals
8. **Vague**: Customer left confused about value

---

## 🎯 Using the Rubric

### **For Individual SAs**
1. Review your top insights (focus on Critical and Important)
2. Read the "Better Approach" and "Example Phrasing"
3. Watch for similar situations in future calls
4. Practice the recommended alternative approaches

### **For Managers**
1. Look for patterns across multiple calls
2. Use severity to prioritize coaching focus
3. Reinforce strengths (positive feedback matters!)
4. Create learning moments around specific examples

### **For Enablement Teams**
1. Identify common gaps across the team
2. Build training around most frequent issues
3. Create playbooks for better approaches
4. Share excellent examples across the team

---

## 🔄 Continuous Improvement

The rubric evaluates calls, but the goal is **continuous improvement**:

1. **Awareness**: Understand what good looks like
2. **Practice**: Try new approaches on calls
3. **Feedback**: Get evaluated again to measure progress
4. **Refinement**: Iterate based on what works

**Remember**: Every SA has strengths and areas for growth. The goal isn't perfection - it's progress! 🚀

---

## ❓ FAQ

**Q: Why is scoring removed?**
A: Numeric scores can feel arbitrary. Qualitative feedback with specific examples is more actionable.

**Q: Why are timestamps so important?**
A: Allows you to review the exact moment and learn from it in context.

**Q: What if the SA did everything perfectly?**
A: You'll still get insights! There's always room for refinement, even for excellent calls.

**Q: How long should discovery take?**
A: No hard rule, but generally 15-30% of the call should be discovery-focused questions.

**Q: What if the customer doesn't want to answer questions?**
A: That's a technique issue! Good discovery makes customers *want* to open up by asking compelling questions.

---

## 📚 Recommended Resources

- **Command of the Message**: Force Management's sales methodology
- **SPIN Selling**: Classic book on discovery questions
- **The Challenger Sale**: Approach to customer engagement
- **Solution Selling**: B2B sales methodology

---

**Need help improving in a specific area?** Review the "Better Approach" sections in your analysis reports - they're tailored to your specific call!
