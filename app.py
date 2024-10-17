import os
import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image as Img, Image
from skimage import io
from torchvision import transforms
from torch.utils.data import DataLoader
import streamlit as st
from model import U2NETP  
from dataloader import RescaleT, ToTensorLab, SalObjDataset  

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    return (d - mi) / (ma - mi)

# Save the output image
def save_output_with_transparency(image_name, pred, d_dir):
    original_image = Image.open(image_name).convert("RGBA")
    predict = pred.squeeze().cpu().data.numpy()
    
    mask = Image.fromarray((predict * 255).astype(np.uint8)).convert("L")
    mask = mask.resize((original_image.size), resample=Image.BILINEAR)
    
    empty_background = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
    result_image = Image.composite(original_image, empty_background, mask)

    result_path = os.path.join(d_dir, os.path.basename(image_name).split(".")[0] + '_result.png')
    result_image.save(result_path)
    return result_path

def process_image(image_path, model):
    test_salobj_dataset = SalObjDataset(
        img_name_list=[image_path], 
        lbl_name_list=[], 
        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    )
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    model.eval()
    with torch.no_grad():
        for data in test_salobj_dataloader:
            inputs_test = data['image'].type(torch.FloatTensor)
            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)
            d1, _, _, _, _, _, _ = model(inputs_test)
            pred = normPRED(d1[:, 0, :, :])
            return pred

# Main function to run Streamlit app
def main():
    # Use a light theme and improve UI with a sidebar for better user experience
    st.set_page_config(page_title="Arka Plan Kaldırma Uygulaması", page_icon="🌄", layout="centered")
    
    st.markdown(
        """
        <style>
        .main { background-color: #f9f9f9; padding: 20px; border-radius: 10px; }
        .stButton>button { border-radius: 5px; background-color: #4CAF50; color: white; }
        .stDownloadButton>button { background-color: #FF5722; color: white; border-radius: 5px; }
        </style>
        """, unsafe_allow_html=True
    )

    st.title("🖼️ Arka Plan Kaldırma Uygulaması")
    st.write("Yüklediğiniz görseldeki arka planı kaldırın ve şeffaf bir arka planla indirip kullanın!")
    
    uploaded_file = st.file_uploader("Bir görüntü seçin...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        with open("uploaded_image.png", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image("uploaded_image.png", caption="Yüklenen Görüntü", use_column_width=True)

        model = U2NETP(3, 1)
        model_dir = 'u2netp.pth'
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_dir))
            model.cuda()
        else:
            model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
        
        if st.button("Arka Planı Kaldır"):
            with st.spinner('🔄 İşleniyor...'):
                pred = process_image("uploaded_image.png", model)
                result_path = save_output_with_transparency("uploaded_image.png", pred, "./")
                
                st.image(result_path, caption="Arka Planı Kaldırılmış Görüntü", use_column_width=True)
                
                with open(result_path, "rb") as file:
                    st.download_button(
                        label="Görseli İndir 📥",
                        data=file,
                        file_name=os.path.basename(result_path),
                        mime="image/png"
                    )

if __name__ == "__main__":
    main()
