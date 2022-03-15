# Handwritten-English-Text-Recognition
## 1.研究動機
因為對於上課的時候，教授在講解ppt或是在黑板上寫重點的時候，偶爾會來不及作筆記或是不能即時理解做成筆記，會拿起自己的手機把黑板上的字跟ppt拍起來，但很常會懶得做成筆記，直接看手機的照片複習，為了解決這個問題，想要做出能夠辨識照片中的文字並能夠自動根據位置排版變成電腦中的文件
## 2.設計原理、研究方法與步驟
Preprocess設計:
  1. 用four-point transformation去偵測圖片文字的邊緣並將圖片進行灰階化
  2. 利用光補償的方式增加文字的可辨識度
  3. Deslanting Algorithm去除草書書寫格式(變成正常、非斜體字型)
  4. Local Adaptive Binarization來增加文字的可讀性
Line Segmentation設計:
  利用OTSU Threshold and Binarization，利用Y軸直方圖投影輪廓與adaptive threshold between valleys來獲取初始線段，運用line drawing algorithm來增加初始線段的正確率
Word Segmentation 設計:
  利用 scale space technique的方式來進行字與字間的切割
Picture Size Standardize 設計:
  使用Python內的OS Library抓取Word Segmentation處理完各個資料夾內的data，接著使用Python Image Library(PIL)內的Image讀檔寫檔功能改寫Image的大小及比例
Word Recognized設計:
  輸入固定是128x32size的圖片，先用5層CNN layers來把圖片中重要的32x256個feature抓出來，接著使用2層RNN layers生成一個32x80的字串，再使用CTC從32x80的字串中找出最佳解與loss value
檔案輸入與輸出:
  使用Python內的OS Library抓取Picture Size Standardize處理完的data，並分成每個Line資料夾中的Word變成分別一行句子
## 3. 系統實現與實驗
  系統:Ubuntu 18.04.4 LTS
  GPU: GeForce RTX 2080 Ti
  程式語言:Python3.7
  開發平台:Tensorflow 1.15.0
## 4. Accuracy of Word Recognition : 88.8283%
## 5. 結論
  成功地將影像中的文字辨識出來並儲存成文字檔，在Preprocess的部分雖然能有效的降低後續切割與辨識文字的時候的誤差



