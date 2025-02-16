#!/usr/bin/env python3

import os
import cv2
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from groq import Groq
from PIL import Image as PILImage
import google.generativeai as genai
import re
import yaml


class AIAssistant(Node):
    def __init__(self):
        super().__init__('ai_assistant')

        # Suppress OpenCV warnings
        os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

        # Load configuration from YAML file
        config_path = "/<YOUR_FILE_LOCATION>/key_prompt.yaml"
        self.get_logger().info(f"Checking config file at: {config_path}")

        if not os.path.exists(config_path):
            self.get_logger().error(f"Config file not found at: {config_path}")
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Load API keys and system messages from config
        groq_api_key = config.get('groq_api_key', 'your_groq_api_key_here')
        genai_api_key = config.get('genai_api_key', 'your_genai_api_key_here')
        self.sys_msg = config.get('sys_msg', 'Default system message')
        self.vision_prompt_text = config.get('vision_prompt', '')

        # Initialize APIs
        self.groq_client = Groq(api_key=groq_api_key)
        genai.configure(api_key=genai_api_key)

        # ROS 2 Communication
        self.publisher_ = self.create_publisher(String, 'ai_response', 10)
        self.subscription = self.create_subscription(String, 'user_input', self.user_input_callback, 10)
        self.bridge = CvBridge()
        self.image_publisher = self.create_publisher(Image, 'webcam_image', 10)

        # Initialize webcam
        self.web_cam = cv2.VideoCapture(0)
        if not self.web_cam.isOpened():
            self.get_logger().error('Error: Could not open webcam.')
            raise RuntimeError('Error: Could not open webcam.')

        self.concvo = [{'role': 'system', 'content': self.sys_msg}]

        # Initialize Google Generative AI model
        self.model = genai.GenerativeModel(
            'gemini-1.5-flash-002',
            generation_config={'temperature': 0.7, 'max_output_tokens': 4096}
        )

        self.latest_frame = None
        self.running = True

        # Start video capture in a separate thread
        self.video_thread = threading.Thread(target=self.live_video_feed, daemon=True)
        self.video_thread.start()

    def groq_prompt(self, prompt, img_context=None):
        if img_context:
            prompt = f'USER PROMPT: {prompt}\n\nIMAGE CONTEXT: {img_context}'
        
        self.concvo.append({'role': 'user', 'content': prompt})
        chat_completion = self.groq_client.chat.completions.create(
            messages=self.concvo, model='llama3-70b-8192'
        )
        response = chat_completion.choices[0].message
        self.concvo.append(response)
        
        # Remove markdown formatting
        cleaned_response = re.sub(r'\*.*?\*', '', response.content)
        return cleaned_response

    def vision_prompt(self, prompt, frame):
        # Convert OpenCV BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(frame_rgb)

        full_prompt = f"{self.vision_prompt_text}\nUSER PROMPT: {prompt}"
        response = self.model.generate_content((full_prompt, pil_img))
        return response.text

    def live_video_feed(self):
        while self.running:
            ret, frame = self.web_cam.read()
            if not ret:
                self.get_logger().error('Failed to capture an image from the webcam.')
                break
            self.latest_frame = frame
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.image_publisher.publish(ros_image)

    def split_into_sentence_chunks(self, text, chunk_size=200):
        words = text.split()
        chunks = []
        current_chunk = ""

        for word in words:
            if len(current_chunk) + len(word) + 1 > chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                current_chunk += " " + word

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def user_input_callback(self, msg):
        prompt = msg.data

        # Use latest frame from live video feed if available
        vision_response = self.vision_prompt(prompt, self.latest_frame) if self.latest_frame is not None else ""
        combined_prompt = f"{prompt}\n\nVision Analysis: {vision_response}" if vision_response else prompt

        response = self.groq_prompt(combined_prompt)
        self.get_logger().info(f"Full AI Response: {response}")

        # Split and publish response in smaller chunks
        chunks = self.split_into_sentence_chunks(response, chunk_size=180)
        for i, chunk in enumerate(chunks):
            chunk_msg = f"{i+1}/{len(chunks)}: {chunk}"
            self.publisher_.publish(String(data=chunk_msg))
            self.get_logger().info(f"Published chunk {i+1}/{len(chunks)}")

    def shutdown(self):
        self.running = False
        self.web_cam.release()
        cv2.destroyAllWindows()
        self.get_logger().info('Shutting down AI assistant.')


def main(args=None):
    rclpy.init(args=args)
    assistant = AIAssistant()
    try:
        rclpy.spin(assistant)
    except KeyboardInterrupt:
        assistant.shutdown()
    finally:
        assistant.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
