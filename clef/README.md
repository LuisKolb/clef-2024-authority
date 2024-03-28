# CLEF checkthat! task 5 Luis Kolb thesis package

## data flows

---

input data format:
- list of dicts with keys:
    - id
    - label
    - rumor
    - timeline
        - author account id
        - tweet link
        - tweet text
    - evidence
        - author account id
        - tweet link
        - tweet text

---

output format to save as TREC-formatted .txt file:
- list of lists with the semantic meaning:
    - rumor_id
    - authority_tweet_id
    - rank
    - score
- also needs a tag

---

input files to verification:
- input data set:
    - list of dicts with keys:
        - id
        - label
        - rumor
        - timeline
            - author account id
            - tweet link
            - tweet text
        - evidence
            - author account id
            - tweet link
            - tweet text
- TREC formatted .txt judgement file

---

output format after verification:
- list of dicts with the keys
    - id
    - label
    - claim
    - predicted_label
    - predicted_evidence, a list with the semantic meaning:
        - author_account
        - tweet_id
        - evidence_text
        - confidence