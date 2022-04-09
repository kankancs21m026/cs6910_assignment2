

**Guide to Execute Code**
- It is **recommanded** to run all the code in **Google Colab**.Please upload the python notebook files available under **Notebook** subdirectory under **PART_A_B** directory.For more tutorial related to Google colab usage follow the link: [Google Colab](https://colab.research.google.com/)


cd C:\DL Assignment2\final\cs6910_assignment2\PART A

python runCNN.py --optimizer "Adam" --lr "0.0001" --dropout "0.2" --image_size "224" --batchNormalization "True" --epoch "2" --filterOrganization "config_all_same" --activation_function "relu" --number_of_neurons_in_the_dense_layer "256" --augment_data "True" 



python runCNN.py --optimizer "Adam" --lr "0.0001" --dropout "0.2" --image_size "224" --batchNormalization "True" --epoch "2" --filterOrganization "custom" --activation_function "relu" --number_of_neurons_in_the_dense_layer "256" --augment_data "True" --no_of_filters "64,64,64,64,64"
