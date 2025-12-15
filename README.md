# SpanPatchTST
This is the official implementation of our paper : "SpanPatchTST:  SpanPatchTST: A Self-Supervised Learning Model for Short-term  and Mid-term Energy Load Forecasting" 

## step-by-step guidelines to reproduce the results

## install dependencies
put all the github repo in a single folder and install the packages listed in the main file ssl.py, lines 18-45

## Reproducibility of the results
In ssl.py, lines 867-893, you should keep the following hyperparameters with the same values as in the code: <br>
seq_len=96, d_model=128, n_heads=16, n_layers=2, weight_decay=0.05, K=32.<br>
Note: hidden_lstm, layer_lstm, and dropout_lstm have no influence on our model (our model does not contain any lstm layer). They were used when we tried to see the effect of adding an lstm encoder. <br>
The remaining hyperparameters whose values are changed throughout the expriments are the following:<br>
*patch_len: each input time series of length 96 is divided into patches, where the patch size = patch_len.
*d_ff: the number of neurons in the FeedForward Network (FFN).<br>
*dropout: the dropout probability of the dropout function to apply on the input embedding result.<br>
*lr: learning rate. <br>
*batch size: batch size. <br>
*mask_ratio: the proportion of patches in the input series to be masked during pretraining. <br>
*attn_dropout: the dropout probability of the dropout function to apply on the attention weights. <br>
*act: activation function in the FFN. <br>
*pre_norm: a boolean variable that indicates whether to apply pre-normalization in the transformer encoder.<br>
*overlap_ratio: the inverse of overlap_ratio represents the proportion of an input patch to be overlapping with an adjacent patch during fine-tuning. <br>
*pt_epoch: the number of pretraining epochs.<br>
*ft_epoch: the number of finetuning epochs. <br>
*pred_len: the prediction length. <br>
*dn: the dataset. Four datasets are used in our paper 'etth1', 'etth2', 'ettm1', and 'ettm2'. For more information about these datasets, you can refer to the [Informer paper's Github repository](https://github.com/zhouhaoyi/Informer2020).<br>

## Training and expected result
Once you set the hyperparameters' values, you run ssl.py. You will see the evolution of the pretraining loss (train vs validation), finetuning loss (train vs validation). Finally, you will get the following:<br>
*Test MSE, Test MAE. <br>
*The weights of the best performing model will be saved in 'best_config_predlen_{pred_len}_{dn}.txt'.<br>
*The visualization of test actual data vs prediction will be displayed and saved as '{dn}_PRED_LEN_{pred_len}.png'.<br>
*Execution time. <br>

## Acknowledgement
Through our work, we tried to improve the results of the paper [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730). We truly appreciate the authors' [Github repository](https://github.com/yuqinie98/PatchTST?tab=readme-ov-file) for the valuable code base. <br>

## 📬 Contact
If you have questions or encounter issues, please [open an issue](https://github.com/lear-ner97/SpanPatchTST/issues) or contact us at **sami DOT benbrahim AT mail DOT concordia DOT ca**.
