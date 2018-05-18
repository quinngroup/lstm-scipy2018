:author: Christian McDaniel
:email: clm121@uga.edu
:institution: University of Georgia

:author: Shannon Quinn, PhD
:email: spq@uga.edu
:institution: University of Georgia

--------------------------------------------------
Developing an LSTM Pipeline for Accelerometer data
--------------------------------------------------

.. class:: abstract

.. class:: keywords

Introduction
------------

As the uses and applications of wearable accelerometry devices for human activity recognition (HAR) have expanded, so too have the corresponding machine learning methods evolved in efforts to improve performance for these tasks. HAR - a term spawned from earlier accelerometer research (Tulenetal97, Bussmanetal98) – is a time series classification problem in which a classifier attempts to discern distinguishable features from movement-capturing on-body sensors (KimCook2010). Typical sensors record changes in velocity through time in the x- y- and z-directions, i.e., accelerometers.
Improvements in and increased accessibility to specialized wearable sensors and sensor-equipped handheld devices, e.g., smartphones, have led to a large number of publicly available datasets and a multitude of experiments and applications using this data.

Accelerometer output consists of high-frequency (30-200Hz) triaxial time series recordings, often containing noise, imprecision, and missing data (Ravietal2005, BaoIntille2004). Furthermore, periods of non-activity – commonly referred to as the Null class – typically represent a large majority of HAR-related accelerometer data (OrdonezRoggen2016). Consequently, attempts to use traditional classifiers have resulted in an apparent upper bound on performance and typically require significant preprocessing and technical engineering of hand crafted features from raw data (Gaoetal2016, Charvarriagaetal2013, ManniniSabatini2010, Gjoreskietal2016, Ravietal2005, OrdonezRoggen2016).

In response to the data complexity and limitations of classical methods, in conjuncture with concurrent theoretical and practical advancements in artificial neural networks, HAR experiments have increasingly made use of non-linear neural models over the last decade. While convolutional neural networks (CNNs) are attractive for their automated feature extraction capabilities during convolution and pooling operations (Sjostrom2017, Rassemetal2017, Fiterauetal2016, SeokKimPark2018, Zebinetal2017, Gaoetal2016, Zhuetal2017, OrdonezRoggen2016, Gjoreskietal2016), recurrent neural networks (RNNs) are specifically designed to extract information from time series data due to the recurrent nature of their data processing and weight updating operations (WilliamsZipser1989). Furthermore, whereas earlier implementations of RNNs suffered from vanishing and exploding gradient problems during training, the incorporation of a multi-gated memory cell in long short-term memory recurrent neural networks (LSTMs) (HochreiterSchmidhuber1997) as well as an upper constraint known as gradient clipping (Pascanuetal2013) helped alleviate these issues.

The promising intuition behind LSTMs motivated researchers in various fields to take on the challenge of training these networks for their data modeling and analysis tasks (SukhwaniPanwar2016, Wooetal2016, Zhaoetal2016, Gersetal2002, Pigouetal2016, Gravesetal2013). As their use increased, numerous studies emerged that addressed various aspects of these complex models and the vast architectural and hyperparameter combinations that are possible (Gersetal2002, ReimersGurevych2017, PressWolf2017, Karpathyetal2015, Merityetal2017). Unfortunately, these pioneering studies tend to focus on tasks other than HAR, e.g., language modeling tasks – which aim to predict the next letter or word in a sequence. This has left the time series classification tasks of HAR without domain-specific architecture guidance or insights into the models’ representation of the data. Consequently, as is discussed in detail below, pilot studies using LSTMs to classify accelerometer data have borrowed what they could from the use of LSTMs in other domains and used their best judgement for the remaining issues.

In a meta-analysis style overview of the use of LSTM RNNs for HAR experiments (discussed below), we found a general lack of consensus or justification regarding the various model architectures and hyperparameters used as well as the overall data analysis pipelines employed. Significant sources of data leakage, where details from the testing data are exposed to the model during training, seem to be largely overlooked. Various preprocessing techniques and unique architectural blueprints have been devised in the absence of established theories for the inner representations of accelerometer data by baseline LSTM models. Without clear justifications for model implementations and deliberate, reproducible data analysis pipelines, objective model comparisons and inferences from results cannot be made. Furthermore, a deeper look into the ways in which these models represent various aspects of the data will help researchers understand the limitations of LSTMs for HAR using accelerometer data, develop novel architectures aimed at solving specific performance limitations, and compare LSTMs with other network archetypes such as gated recurrent unit RNNs (GRUs) and more advanced frameworks such as attention-based neural networks, neural Turing Machines, and memory augmented neural networks.

In this paper we survey the various preprocessing methods and LSTM architectures implemented in the field in a meta-analysis style. Additionally, we explore ways to characterize both the class-wise raw accelerometer data and the post hoc LSTM parameter embeddings for network insights. Finally, we discuss the data preparation, training procedure, and performance evaluation stages of data analysis so as to establish a thorough and reproducible data analysis pipeline. Our pipeline and discussion points are demonstrated using various open-source benchmark accelerometer datasets. We suspect that such efforts will provide unique insights into the usefulness of LSTMs for classifying accelerometer data and will allow for scientifically rigorous comparisons across experiments and datasets.

Related Works
-------------

*Human Activity Recognition*

HAR experiments first took place in the biomedical clinic in the late 1990’s and early 2000’s (Bussmanetal2001, Ravietal2005, Tulenetal97, Bussmanetal98), and quickly spread, leading to HAR research in current-day innovative settings such as the automobile (Carvalhoetal2017), the bedroom (Moreauetal2016), the dining room (Kyritsisetal2017), and outdoor sporting environments (Rassemetal2017).

LSTM models for HAR tasks – a Meta-study

For this paper, a meta-analysis style survey of the use of LSTM RNNs for HAR tasks using accelerometer data was conducted using 27 studies. Published works as well as pre-published and thesis research projects were included so as to gain insight into the state-of-the-art methodologies at all levels and increase the volume of works available for review. It should be noted that the following summaries are not necessarily entirely exhaustive in the specifications listed or the individual citations made for each specification. Additionally, many reports did not include explicit details of many aspects of their research.

*Experimental Setups*

Across the 27 studies, each used a unique implementation of LSTMs for the research conducted therein. Many reports used the open-source OPPORTUNITY Activity Recognition dataset (Roggenetal2010, OrdonezRoggen2016, Riveraetal2017, Gaoetal2016, Zhaoetal2017, Broome2017, GuanPlotz2017), while other datasets include PAMAP2 (OrdonezRoggen2016, Setterquist2018, GuanPlotz2017, Zhangetal2018), Skoda (OrdonezRoggen2016, GuanPlotz2017), WISDM (Chenetal2016, U2018), ChaLearn LAP large-scale Isolated Gesture dataset (IsoGD) (Zhangetal2017), Sheffield Kinect Gesture (SKIG) dataset (Zhangetal2017), UCI HAR dataset (U2018, Zhaoetal2017), a multitude of fall-related datasets (Muscietal2018), and various study-specific internally-collected datasets. Programming packages used include Theano Lasagne, RNNLib, and Keras with TensorFlow. While most of the studies we examined trained models on tasks under the broad umbrella of “Activities of Daily Life” (ADL) – e.g., opening a drawer, climbing stairs, walking, or sitting down – several of the studies focused on more specific human activities such as smoking(…), cross-country skiing (Rassemetal2017), eating (Kyritsisetal2017), nighttime scratching (Moreauetal2016), and driving (Carvalhoetal2017).

Numerous experimental data analysis pipelines were used, including cross validation (Lefebvreetal2015), repeating experiments (ShinSung2016), and various train-validation-test splitting procedures (Sjostrum2017, WuAdu2017, Huetal2018). Of note, several studies split data from the same participant between training and testing data (Huetal2018), which we have pinpointed as a potential source of data leakage, or at least an approach less emulative of real-world situations in which the “testing” data will consist of data from newly encountered individuals.

*Preprocessing*

Before training the proposed models, each study performed some degree of preprocessing. Some reports kept preprocessing to a minimum, e.g., linear interpolation to fill missing values (OrdonezRoggen2016), per-channel normalization (OrdonezRoggen2016, Gaoetal2016, Huetal2018), and simple standardization (Chenetal2016, Zhaoetal2017, Moreauetal2016). Typically, data is standardized to have zero mean, i.e., centering the amplitude around zero (Broome2017), and unit standard deviation, whereas (Zhaoetal2017) standardized the data to have 0.5 standard deviation, citing (…) as supporting this nuance for deep learning implementations. Standardization is often important for data-dependent models such as LSTM RNNs since the presence of outliers and skewed distributions may distort the weight embeddings (…). Furthermore, if the common sliding window technique is used (discussed further below), standardization can be utilized for online activity classification. For these reasons, we standardize the inputs to our models in this experiment.

Other noise reduction strategies employed include kernel smoothing (Gaoetal2016), removing the gravity component (Moreauetal2016), applying a low-pass filter (Lefebvreetal2015), removing the initial and last 0.5 seconds (Huetal2018), and normalizing the length of each gesture by down sampling (…). Gao, et. al. go so far as to apply Nadaraya-Watson kernel weighted average smoothing, using the Epanachnikov quadratic kernel and 40-nearest neighbor window size (Gaoetal2016). Moreau, et. al. used the derivative of the axis-wise gravity component in order to group together segments of data from different axes, tracking a single motion across axes as the sensor rotated during a gesture (Moreauetal2016).

While these methods are not exceedingly technical or difficult to implement, they do require a degree of domain knowledge in signal processing, and are more computationally expensive and less realistic for online and on-device implementations than is desired. Much of the appeal of non-linear models such as neural networks is their ability to learn from raw data itself and independently perform smoothing and feature extraction on noisy data, so we aim to keep preprocessing to a minimum in our experiments and instead rely on the models themselves.

Some form of data redistribution or organization was also typical. For example, Broome 2017 and Moreau, et. al. excluded the dominant Null class as a solution to class imbalance; however, this is not very feasible for real-world online activity classification, where long periods of non-activity between meaningful segments are to be expected. Lee & Cho aimed to circumvent the Null-related class imbalance by first training a model to differentiate meaningful data segments from the Null class, and subsequently training a second model to predict the specific gesture class (LeeCho2013). Moreau, et. al. used resampling to solve class imbalance.

For feeding the data into the models, the sliding window technique was commonly used, with vast discrepancy in the optimal size of the window (reported both as units of time and number of timepoints) and step size. Some of the window sizes used include 30 (Broome2017), 50 (Chenetal2016, 20), 80 (WuAdu2017), and 100 (Zhaoetal2016) timepoints, and 32 (Muscietal2018), 250 (Bergelin2017), 500 (OrdonezRoggen2016,WuAdu2017), 1000 (Rassemetal2017,Sjostrum2017), 2000 (Riveraetal2017), 3000 (Moreauetal2016), 4000 (Huetal2018), and 5000 (Zhaoetal2017) milliseconds (ms). Step sizes between windows include 250 (OrdonezRoggen2016) and 1000 (Huetal2018) ms, or an adaptive percentage of the window length, such as 50% (Rassemetal2017, Sjostrum2017, Broome2017). Finally, Guan & Plotz ran an ensemble of models, each using a random sampling of a random number of frames with varying sample lengths and starting points using a wrap-around windowing method. This method is similar to the bagging scheme of random forests and was implemented to increase robustness of the model.

Once a window is generated it must be assigned a class and labeled as such. Labeling schemes used include a jumping window technique, where the class of the last data point in the window is used as the class label (OrdonezRoggen2016) or using the majority class within the window (Broome2017).

At this point we reiterate that we saw no explicit evidence of efforts to prevent data leakage during preprocessing. Data leakage occurs when any smoothing, grouping, filtering, or other operations are performed on the entire dataset beforeseparating the test set. If any preprocessing is to be performed on the test set, only parameters from the training set can be used. For example, when standardizing the testing set, the researcher should first separately standardize the training set and then use the mean and standard deviation of the training set as parameters for standardizing the test set. If possible, test set data should come from different participants or even different datasets than those used for the training data.

*Architectures*

Numerous different architectural and hyperparameter choices were made among the various studies. Most studies used two LSTM layers (OrdonezRoggen2016, Chenetal2016, Kyritsisetal2017, Zhangetal2017, Riveraetal2017, U2018, Zhaoetal2017, GuanPlotz2017, Huetal2018, Muscietal2018), while others used a single layer (WuAdu2017, Broome2017, ShinSung2016, Carvalhoetal2017, Zhaoetal2016, Zhangetal2018, Seoketal2018) or three layers (Zhaoetal2016). The choice for two layers seems to have generated from (…), which reports that at least two layers seemed to be adequate for the problem in that study, but the generalizability of that statement is questionable.

Several studies designed or utilized novel LSTM architectures that went beyond the simple tuning of hyperparameters. Before we list them, note that the term “deep” in reference to neural network architectures indicates the use of multiple layers of hidden connections; for LSTMs, an architecture generally qualifies as “deep” if it has three or more hidden layers. These include the combination of CNNs with LSTMs such as ConvLSTM (Zhangetal2017, Gaoetal2016) and DeepConvLSTM (OrdonezRoggen2016, Sjostrum2017, Broome2017); innovations related to the connections between hidden units including the bidirectional LSTM (b-LSTM) (Rassemetal2017, 5, Broome2017, Moreauetal2016, Lefebvreetal2015), hierarchical b-LSTM (LeeCho2012), deep residual b-LSTM (Zhaoetal2017), and LSTM with peephole connections (p-LSTM) (Rassemetal2017); and other nuanced architectures such as ensemble deep LSTM (GuanPlotz2017), weighted-average spatial LSTM (WAS-LSTM), deep-Q LSTM (Seoketal2018), and similarity-based LSTM (Fiterauetal2016). The use of densely-connected layers before or after the LSTM layers was also common. Zhang, et. al. used three dense layers before the LSTM layers (Zhangetal2018), Kyritsis, et. al. added a dense layer with ReLU activation after the LSTM layers (Kyritsisetal2017), Zhao, et. al. included a dense layer with tanh activation after the LSTMs, and Musci, et. al. used a dense layer before and after its two LSTM layers (Zhaoetal2016, Muscietal2018). Both the deep-Q LSTM and the similarity-based LSTM used a combination of dense and LSTM hidden layers.

Once the number of layers is determined, the number of units per LSTM layer must be set. The number of units per layer specified by various studies include 3 (Moreauetal2016), 14 (Zhaoetal2016), (WuAdu2017), 28 (Zhaoetal2017), 32 (Muscietal2018), 64 (Huetal2018), 100 (Lefebvreetal2015), 128 (OrdonezRoggen2016, Sjostrum2017, Zhaoetal2017, ShinSung2016), 256 (Riveraetal2017,GuanPlotz2017), and 512 (Setterquist2018). Several studies used different numbers of units for different circumstances – e.g., three units per layer for unilateral movement (one arm) and four units per layer for bilateral movement (both arms) (Moreauetal2016) or 28 units per layer for the UCI HAR dataset (lower dimensionality) versus 128 units per layer for the Opportunity dataset (Zhaoetal2017). Others used different numbers of units for different layers of the same model – e.g., 14-14-21 for a 3-layer model (Zhaoetal2016).

Many studies tested multiple options for the number of units per layer, exemplifying a theme throughout the studies: hyperparameter ranges or sets were used by specific studies that largely or entirely do not overlap with the ranges or sets used by other studies. For example, (…) assessed the performance of models with two, five, or ten units per layer, Rassem, et. al. constructed models with 25, 35, or 50 units per layer (Rassemetal2017), and Setterquist 2018 searched from 8-512 units per layer.

Almost all of the reports used the tanh activation function for their LSTM cell outputs as this is the activation function used the original paper (HochreiterSchmidhuber1997), but others used include ReLU (Zhaoetal2017, Huetal2018) and sigmoid (Zhangetal2018).

*Training*

Once a model architecture is specified, it must be trained and the weights must be updated through a back propagation technique developed specifically for recurrent neural networks known as back-propagation through time (BPTT). Weights are often initialized using specific strategies, for example random orthogonal initialization (OrdonezRoggen2016, Sjostrum2017), fixed random seed (Setterquist2018), the Glorot uniform initialization (Broome2017), random uniform initialization within [-1, 1] (Moreauetal2016), or using a random normal distribution (Huetal2018). Training may occur using all the input data at once, or in mini-batches of examples of size 32 (Riveraetal2017, Setterquist2018), 100 (OrdonezRoggen2016, Chenetal2016, Sjostrum2017), 200 (Huetal2018), or 450 (Bergelin2017).

To calculate the amount of change needed for each training epoch, different loss functions are used. Categorical cross-entropy is the most widely used method (OrdonezRoggen2016,Chenetal2016,Sjostrum2017,Kyritsisetal2017,Setterquist2018,Broome2017,19,Huetal2018, Zhangetal2018), but F1 score loss (19), mean squared error (MSE) (Carvalhoetal2017), and mean absolute error and root MSE (Zhaoetal2016) were also used with varying degrees of success. During back propagation, various updating rules – e.g. RMSProp (OrdonezRoggen2016, Setterquist2018, Broome2017), Adam (Kyritsisetal2017, Broome2017, 19, Huetal2018, Zhangetal2018), and Adagrad (ShinSung2016) – and learning rates – 10-7(ShinSung2016), 10-4(Sjostrum2017, GuanPlotz2017), 2e-4(Moreauetal2016), 5e-4 (Lefebvreetal2015), and 10-2(OrdonezRoggen2016) are used.

Regularization techniques are often employed to stabilize the weight update process and avoid the problem of exploding gradients (LSTMs are not susceptible to vanishing gradients (HochreiterSchmidhuber1997). Regularization techniques employed include weight decay of 0.9 (OrdonezRoggen2016,Sjostrum2017); update momentum of 0.9 (Moreauetal2016), 0.2 (Lefebvreetal2015), or the Nesterov implementation (ShinSung2016); dropout (forgetting the output from a proportion of units, e.g., 0.5 (OrdonezRoggen2016,Sjostrum2017) or 0.7 (Zhaoetal2016)) between various layers; batch normalization (Zhaoetal2017); or gradient clipping using the norm (Zhaoetal2017, Huetal2018, Zhangetal2018). 1Broome 2017 chose to use the stateful configuration for its baseline LSTM. In this configuration, unit memory cell weights are maintained between each training example instead of resetting them to zero after each forward pass (…).

Finally, models are trained for a given number of iterations, i.e., epochs. The number of epochs specified ranged from 100 (Broome2017) to 10,000 (Huetal2018). Many studies chose to use early stopping, which stops training once performance on the validation set has slowed or halted. This prevents overfitting, which occurs when the model learns to represent irreproducible error in the training data (…). Various patience schemes, specifying how many epochs the model should allow with no improvement above a given threshold, were chosen, including 10, 20, and 50 epochs.

*Performance measures*

Once the model has been trained, it is given a set of examples it has not seen yet and makes predictions on the target class that each example belongs to. Various performance measures are used to assess the performance of the model on this test set. These measures include the F1 score used by most (OrdonezRoggen2016, Broome2017, Gaoetal2016, Zhaoetal2017, Broome2017), classification error (Rassemetal2017), accuracy (Sjostrum2017, Setterquist2018), and ROC (Moreauetal2016, Huetal2018). The use of different performance measures makes comparisons across studies difficult.

As this meta-analysis style overview has shown, there are many different model constructions being employed for HAR tasks. The lack of clear understanding for how the LSTM layers are representing this specific data and which hyperparameter choices may be better for specific problems within the field has motivated the current study.

Experimental Setup
------------------

Datasets

Many studies use the high-dimensional data from inertial sensors, which supplement accelerometer measurements with axis-wise rotation information via gyroscopes and axis-wise changes in the surrounding magnetic field via magnetometers. However, accelerometer data is ubiquitous in this field and the decreased feature space has the benefits of illuminating the robustness of classification methods used in addition to lower computational complexity, making on-line and on-device classifications more feasible. As such, this report mainly focuses on the use of triaxial accelerometer data.
-      List datasets

*Preprocessing*
-      standardization

*Training*
-      CV, hyperparameter ranges and tuning (heuristics based)

*Performance Measures*
-      F1

*Model Exploration*
-      Visualize the weight embeddings and find patterns with specific classes (activities)

Results
-------

Discussion/Future Work
----------------------

Conclusion
----------
