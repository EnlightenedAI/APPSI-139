# APPSI-139: A Corpus of English Application Privacy Policy Summarization and Interpretation.

## introduction
The privacy policy is an important statement or agreement for users to understand how service providers collect, process, store, and use personal information. However, privacy policies are usually excessively long and filled with complex technobabble and legalese, leading most ordinary users to bypass reading by clicking on ''Agree'' and ''Join Now''. Regrettably, this behavior may expose users to privacy risks due to ignoring privacy policies. In this paper, we address this contradiction by building high-quality privacy policy corpora and developing a framework for privacy policy summarization and interpretation. Initially, we build an English privacy policy corpus APPSI-139, meticulously annotated by legal experts, which contains 139 English application privacy policies, 15,692 rewritten sentences, and 36,351 tags. Furthermore, we present Tcsi-pp-MTL, a refined iteration of the Tcsi-pp framework which is designed for training and inferencing of privacy policy summarization models. The TCSI-pp-MTL framework, integrating end-to-end multi-task learning, maintains the professional capability of privacy policy summarization while reducing computational costs by approximately 80% compared to TCSI-pp. Our experiments demonstrate that the application privacy policy summarization models implemented based on the Tcsi-pp-MTL framework offer superior readability and reliability compared with large language models such as GPT-4o.

<!-- The guide for [Paper](Documents/Annotation_Guidelines_Chinese_Version.pdf), [Appendix](Documents/Annotation_Guidelines_Chinese_Version.pdf) and [Annotation Guidelines](Documents/Annotation_Guidelines_English_Version.pdf)) explains the tags and the process of annotation, which can be found in the Documents.  -->

## environment

Project dependencies can be installed in the following ways:

```
pip install -r requirements.txt
```

### How to use

You can be infer in the following ways:
```
python ./Infer/main.py --topic_list choose_a_topic_list  --data data_name
```
**(New)** The mt5_mtl_model is currently hosted at the following [link](https://huggingface.co/EnlightenedAI/APPSI-139/tree/main). Additionally, we will be uploading all associated model parameters to this same location for easy access and reference.
### Effect Demonstration
Figure1 displays the TCSI-pp-MTL in summarization of privacy policy.

```
![My image](./images/TCSI-pp-MTL.png)
```

## Update
We will continue to update this repository on Github.
