# LAWNet

## Introduction

This repository gives the implementation of our proposed Wrist ROI Extraction method based on Key Vectors and Wrist Vein Features Extraction Network using Saturation channel - LAWNet. We also provides a self-collected RGB wrist database for research on wrist vein recognition in smartphones.

## Database

The NTUST-IW contactless wrist vein database was made of 2120 RGB images captured by an iPhone 8+ (indoor and outdoor).

### Collection Method

We designed and installed an application for wrist vein collection on an iPhone 8+ with the real-time video of the camera capture (640×480 resolution). The collected wrist vein images were then transferred to the secured server for storage. Volunteers used this application to capture their wrist vein images.

### Collection Steps

1. Volunteers were informed of the experiment they will be part of and their rights. They read and signed the informed consent form as an agreement.
2. Registration of the information of the volunteers.
3. Collector demonstrated the database collection application from the volunteers.
4. Volunteers used our provided iPhone 8+ to collect their wrist vein images. First, a volunteer freely presented one wrist out. Then, another hand held the phone with the wrist vein collection application turned on to capture wrist. The collected images were transferred to the secured server for storage.
5. In the first session, 10 samples per person (five samples per wrist) were captured indoor in the laboratory.
6. The collector checked the captured images in the server. If some images were not accepted because of no wrist vein region in images, or too blurred due to shaking, the volunteers were asked to recapture again.
7. One week later, in the second session, 10 samples per person (five samples per wrist) were captured outdoor in the school campus.
8. Step 6 was repeated in the second session. The capturing process was stopped when obtaining enough images.

Finally, ten images were collected from each left and right wrist hand. The database was collected by smartphones with normal cameras in an uncontrolled environment, thus increasing the image variants. This is close to practice use. Figure below illustrates the database collection process.

![Database Collection Process](https://github.com/vantrungfit/lawnet/blob/main/Images/database_collection_process.png)

### Reference Samples

Researchers can refer to 100 samples of NTUST-IW database (10 classes in ['Images/100_samples/'](https://github.com/vantrungfit/lawnet/blob/main/Images/100_samples) folder). Below are wrist samples and corresponding ROIs of four wrists (indoor and outdoor) in NTUST-IW database: (a)Wrist image, (b) Wrist ROI on the saturation channel, (c) Extracted ROI.

<table>
    <thead>
        <tr>
            <th>Individual</th>
            <th>Session</th>
            <th>(a)</th>
            <th>(b)</th>
            <th>(c)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>1</td>
            <td>Indoor</td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s1_ia.png" alt="s1_ia" width = auto height = auto></td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s1_ib.png" alt="s1_ib" width = auto height = auto></td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s1_ic.png" alt="s1_ic" width = auto height = auto></td>
        </tr>
        <tr>
            <td>Outdoor</td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s1_oa.png" alt="s1_oa" width = auto height = auto></td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s1_ob.png" alt="s1_ob" width = auto height = auto></td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s1_oc.png" alt="s1_oc" width = auto height = auto></td>
        </tr>
        <tr>
            <td rowspan=2>2</td>
            <td>Indoor</td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s2_ia.png" alt="s2_ia" width = auto height = auto></td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s2_ib.png" alt="s2_ib" width = auto height = auto></td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s2_ic.png" alt="s2_ic" width = auto height = auto></td>
        </tr>
        <tr>
            <td>Outdoor</td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s2_oa.png" alt="s2_oa" width = auto height = auto></td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s2_ob.png" alt="s2_ob" width = auto height = auto></td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s2_oc.png" alt="s2_oc" width = auto height = auto></td>
        </tr>
         <tr>
            <td rowspan=2>3</td>
            <td>Indoor</td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s3_ia.png" alt="s3_ia" width = auto height = auto></td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s3_ib.png" alt="s3_ib" width = auto height = auto></td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s3_ic.png" alt="s3_ic" width = auto height = auto></td>
        </tr>
        <tr>
            <td>Outdoor</td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s3_oa.png" alt="s3_oa" width = auto height = auto></td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s3_ob.png" alt="s3_ob" width = auto height = auto></td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s3_oc.png" alt="s3_oc" width = auto height = auto></td>
        </tr>
         <tr>
            <td rowspan=2>4</td>
            <td>Indoor</td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s4_ia.png" alt="s4_ia" width = auto height = auto></td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s4_ib.png" alt="s4_ib" width = auto height = auto></td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s4_ic.png" alt="s4_ic" width = auto height = auto></td>
        </tr>
        <tr>
            <td>Outdoor</td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s4_oa.png" alt="s4_oa" width = auto height = auto></td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s4_ob.png" alt="s4_ob" width = auto height = auto></td>
            <td><img src="https://github.com/vantrungfit/lawnet/blob/main/Images/ntust_iw_s4_oc.png" alt="s4_oc" width = auto height = auto></td>
        </tr>
    </tbody>
</table>

### Dowloading Instruction

To get full of our databases, please follow the [ANNEX-A terms for use of database](https://github.com/vantrungfit/lawnet/blob/main/ANNEX_A_terms_for_use_of_database.docx) and [Database release agreement](https://github.com/vantrungfit/lawnet/blob/main/Database_release_agreement.docx) files.

## Training and Testing

### Prepare databases
In ['Keras_Code\data\\'](https://github.com/vantrungfit/lawnet/blob/main/Keras_Code/data/) folder, put ROI images of each database into the corresponding folder in the following format: 'Database_name\ID\X.Y', where 'ID' is the unique identifier of individual, 'X' is image filename, 'Y' is image extension. For example: "NTUST-IP\0001\1_01.tiff", 'database_name' is 'NTUST-IP', 'ID' is '0001', filename is '1_01', extension is 'tiff'.

The names of databases for training are listed in the  file ['Keras_Code\data\train_folders.txt'](https://github.com/vantrungfit/lawnet/blob/main/Keras_Code/data/train_folders.txt).

The names of databases with the corresponding number of sessions for testing are listed in the  file ['Keras_Code\data\test_folders.txt'](https://github.com/vantrungfit/lawnet/blob/main/Keras_Code/data/test_folders.txt)

For example, if you want to run 'Open-set' testing (the test set is not included in the train set) on 'NTUST-IW' database (two sessions), then in the file ['train_folders.txt'](https://github.com/vantrungfit/lawnet/blob/main/Keras_Code/data/train_folders.txt) you input the following lines:
        
    tongji
    polyu
    casia
    xjtu-up
    ntust-hp
    ntust-ip
    
and in the file ['test_folders.txt'](https://github.com/vantrungfit/lawnet/blob/main/Keras_Code/data/test_folders.txt) you input the line:

    ntust-iw 2
    
where '2' is number of sessions.

### Settings
In the file ['Keras_Code\options.py'](https://github.com/vantrungfit/lawnet/blob/main/Keras_Code/options.py) : the default settings are as used in our paper, you can run different training sessions or models by changing the default values of 'train_session' and 'model_name' options in this file also. You can set the default value of 'train_session' option to be 1 for 'Closed-set' testing (the test set is included in the train set) or 2 for 'Open-set' testing (the test set is not included in the train set). The default value of 'model_name' option is one of the following names:

    cnn1,
    cnn2,
    vgg16,
    mobilenet_v1
    mobilenet_v2
    mobilenet_v3
    mobilefacenet
    mpsnet
    lawnet
### Train and test models
After preparing databases and settings, you can train and test model by running file ['Keras_Code\train.py'](https://github.com/vantrungfit/lawnet/blob/main/Keras_Code/train.py). The results are stored in the folder 'Keras_Code\results\session_X\model_name\\', where 'X' is the training session and 'model_name' is the name of model.
