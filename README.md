# Eye Disease Classification Using Deep Learning 👀

This project aims to develop a deep learning model using Convolutional Neural Networks (CNN) to classify retinal images into four categories: Normal, Diabetic Retinopathy, Cataract, and Glaucoma. The dataset comprises approximately 1000 retinal images for each class, collected from various sources, including IDRiD, Ocular recognition, and HRF. The goal is to leverage CNN’s powerful feature extraction capabilities to accurately diagnose these eye conditions based on retinal images. This model could assist in early detection and treatment planning for patients, potentially improving outcomes for those with retinal diseases.

# Example Images From Each Class
![image](https://github.com/user-attachments/assets/b906926d-ae02-4288-b788-1a5f2520c0e8)


# Data Cleaning and Wrangling Process

The data wrangling process involved multiple stages to prepare and augment the retinal image dataset. First, image sizes were made consistent by resizing all images to a target size of 256x256 pixels. This ensures uniformity for input into the Convolutional Neural Network (CNN) model. Duplicate images were identified and replaced using data augmentation techniques like rotation, zoom, and flips. This helped increase the variability and improve generalization.

Additional data augmentation was applied using the Albumentations library to expand the dataset. The process involved loading specific images from underrepresented classes (e.g., Cataract, Diabetic Retinopathy, Glaucoma, and Normal), applying transformations (like random crops, rotations, and brightness/contrast adjustments), and saving the augmented images in their respective folders. The number of augmented images varied for each class to balance the dataset, ensuring that each category had enough samples for training.

In total, 93 images were augmented for Glaucoma, 62 for Cataract, 2 for Diabetic Retinopathy, and 26 for the Normal class. This extensive augmentation process improved the dataset size and balance, enhancing the model's ability to generalize across all retinal image classes.

# Results and Outcomes 📈🎯

**Model Performance**  <br /> 
The model achieved a validation accuracy of 76.48%, which is decent but indicates room for improvement. The training accuracy improved consistently across the epochs, while the validation accuracy fluctuated significantly, highlighting that the model is likely overfitting to the training data. Additionally, the validation loss remained high, at 0.8261, compared to the training loss, which decreased steadily. The discrepancy between training and validation performance suggests that the model is learning to fit the training data well, but struggles to generalize to unseen data, which is a key indication of overfitting.

**Confusion Matrix**  <br /> 
The confusion matrix provides insight into how well the model performs in classifying the four categories (cataract, diabetic retinopathy, glaucoma, and normal). For each class, there are significant misclassifications. For instance, the model correctly classified 73 normal images, but it misclassified a large number of normal images into other classes (especially cataract and diabetic retinopathy). Similar patterns are seen in the other classes, with many predictions spread across multiple incorrect categories. This highlights that the model is struggling to differentiate effectively between the different disease categories, leading to a relatively low accuracy for each individual class.

**Classification Report**  <br /> 
The classification report further emphasizes the model’s difficulty in distinguishing between classes. Precision, recall, and f1-scores for all four classes range from 0.23 to 0.27, which indicates weak performance across the board. Recall for the "normal" class is slightly higher at 33%, suggesting that the model is slightly better at identifying normal cases, but it still performs poorly overall. The similar performance metrics across all classes show that the model lacks the ability to confidently distinguish between diseases, which leads to a relatively poor classification performance.

**Key Observations**  <br /> 
The model has successfully learned useful features for the training data but struggles with generalization to the validation set, as seen in the fluctuating validation accuracy and high loss values. Overfitting is a key issue here, as the model performs much better on the training data than on the validation data. The confusion matrix and classification report reinforce this, as the model fails to make accurate distinctions between the four classes. The fluctuations in validation accuracy, paired with the low precision and recall, suggest that the model may not have learned enough diverse patterns to effectively generalize to unseen images.

**Next Steps**  <br /> 
To address the performance issues and improve the model's generalization, a few strategies can be explored. First, applying more aggressive data augmentation techniques could introduce more variation in the training images, helping the model learn more generalized patterns. Second, incorporating regularization methods such as L2 regularization or increasing the dropout rate can help reduce overfitting. Additionally, adding more diverse data for each class, if available, could enhance the model's ability to distinguish between different disease categories. Lastly, experimenting with a learning rate scheduler could allow the model to adjust its learning more effectively over time, potentially improving performance on the validation set.
