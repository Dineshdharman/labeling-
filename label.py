import os
import pandas as pd
from PIL import Image

# Directory containing the images
image_dir = r'C:\Users\HP\Desktop\model\uploads'

# Manually labeled data (Example)
labels = {
    '1636.jpg': 'boys shoes',
    '1637.jpg': 'boys shoes',
    '1653.jpg': 'boys shoes',
    '1836.jpg': 'boys shoes',
    '2211.jpg': 'boys slippers',
    '2477.jpg': 'boys shoes',
    '2626.jpg': 'girls slippers',
    '2703.jpg': 'tshirt',
    '2722.jpg': 'shorts',
    '2727.jpg': 'tshirt',
    '3307.jpg': 'boys shoes',
    '3322.jpg': 'tshirt',
    '3324.jpg': 'tshirt',
    '3809.jpg': 'tshirt',
    '3820.jpg': 'girls vest',
    '4113.jpg': 'boys shoes',
    '4170.jpg': 'boys shoes',
    '5441.jpg': 'shirt',
    '6820.jpg': 'girls slippers',
    '6827.jpg': 'girls shoes',
    '6829.jpg': 'boys shoes',
    '8071.jpg': 'boys shoes',
    '9110.jpg': 'boys shoes',
    '9116.jpg': 'boys shoes',
    '9118.jpg': 'boys shoes',
    '9119.jpg': 'boys shoes',
    '9383.jpg': 'boys shoes',
    '10054.jpg': 'tshirt',
    '31097.jpg': 'shirt',
    '31123.jpg': 'girls tshirt',
    '43369.jpg': 'boys slippers',
    '44218.jpg': 'girls slippers',
    '52127.jpg': 'tshirt',
    '56993.jpg': 'girls shoes',
    '58488.jpg': 'girls heels',
    '59943.jpg': 'girls shoes',
    'img1.jpg': 'boys shoes',
    'img2.jpg': 'boys formal shoes'

    # Add more manually labeled images
}

# Save manually labeled data to an Excel file
data = []
for image_name, label in labels.items():
    img_path = os.path.join(image_dir, image_name)
    data.append([image_name, label])

df = pd.DataFrame(data, columns=['Image', 'Label'])
df.to_excel('labeled_images.xlsx', index=False)
