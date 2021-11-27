# SIGNTEGRATE
Vietnamese Sign Language recognition system

keywords: skeleton-based action recognition, human action recognition.

A customized system to recognize Vietnamese sign language. 

Preprocessing video frames -> skeleton data with Mediapipe. 
We use DD-Net (Yang et al https://arxiv.org/pdf/1907.09658.pdf) - a state-of-the-art deep learning model for human action recognition task. 
The system can recognize 10 simple sentences, which are (in vietnamese): 'xin chao rat vui duoc gap ban', 'tam biet hen gap lai', 'xin cam on ban that tot bung', 'toi xin loi ban co sao khong', 'toi yeu gia dinh va ban be', 'toi la hoc sinh', 'toi thich dong vat', 'toi an com', 'toi song o viet nam', 'toi la nguoi diec'. 