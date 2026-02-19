# LLM-based Network Anomaly Detection (6G-AIOps)

##  Project Overview
This project demonstrates the use of **Small Language Models (SLMs)** for autonomous network security. By fine-tuning **Microsoft's phi-2** on network traffic logs, we transform raw numerical data into a "language" that the AI can reason about. This is a foundational step toward **6G Native AI**, where networks are self-healing and self-securing.

---

## The Context: Why 6G and LLMs?

### 1. What is 6G?
**6G** is the future of wireless communication (expected around 2030). Unlike 5G, 6G is designed to be **AI-Native**. This means AI isn't just an add-on; it's the "brain" of the network, managing massive data speeds and billions of IoT devices with micro-second latency.

### 2. How are attacks detected today?
Currently, most systems use:
* **Signature-based detection:** Looking for known "fingerprints" of old viruses. It fails against new, unknown attacks.
* **Traditional Machine Learning:** Using mathematical algorithms (like LSTMs) to find outliers. While fast, they are "black boxes" and cannot explain **why** a certain flow was flagged.

### 3. The LLM Advantage
Using a **Large Language Model (LLM)** allows the system to:
* **Reason:** Understand the context of the traffic patterns, not just the raw numbers.
* **Explain (XAI):** Provide a human-readable report. For example: *"I flagged this as a potential DDoS attack because the packet frequency from this source IP is 100x higher than the baseline for this protocol."*

---

## Dataset: NSL-KDD
This project utilizes the **NSL-KDD dataset**, an industry-standard benchmark for testing Intrusion Detection Systems (IDS).

### Why NSL-KDD?
* It is a refined version of the KDD'99 dataset, removing redundant records that could bias the AI model.
* It covers 4 main attack categories:
    1. **DoS** (Denial of Service)
    2. **R2L** (Remote to Local)
    3. **U2R** (User to Root)
    4. **Probing**

### 🔗 Dataset Link
You can download the dataset from Kaggle:
**[NSL-KDD Dataset on Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd)**

---

