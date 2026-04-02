# Temporal Alignment in Large Language Models

This project studies the problem of **temporal alignment** in Large Language Models (LLMs), i.e., how model behaviour changes when preferences, instructions, or objectives are introduced sequentially over time.

In real-world settings, AI systems are often updated repeatedly (e.g., safety → helpfulness → style). However, sequential alignment may cause **interference, forgetting, or drift** between objectives. This creates challenges in maintaining stable behaviour across multiple alignment goals.

From a psychological perspective, this relates to how systems maintain **consistent behavioural tendencies** while adapting to new information. Similar to human cognition, new learning may modify or override previous behavioural patterns.

Our work aims to better understand:

- how alignment objectives interact over time
- whether new objectives override previous ones
- how stable behavioural patterns can be maintained
- how sequential preference learning affects consistency

## Benchmark focus

We study temporal alignment using sequential preference optimisation setups across:

- harmlessness
- helpfulness
- style / behaviour consistency

Key challenges:

- preference conflict across objectives
- behavioural drift after sequential updates
- instability in learned representations
- difficulty preserving previously learned alignment signals

## Research direction

We explore methods to:

- measure behavioural changes across alignment stages
- analyse interference between objectives
- improve stability of sequential preference learning
- understand alignment dynamics using insights from psychology and behavioural consistency

