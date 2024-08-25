import logging
import requests
import json
import os
import dotenv
import concurrent.futures
import base64
import tempfile
import random
import time
import numpy
from datetime import datetime
from pillow_lut import load_cube_file
from io import BytesIO
from PIL import Image, ImageEnhance

class StellaApp():
    """ikmalsaid"s STELLA AI Studio (Version 24.0721). Copyright (C) 2024 All rights reserved.
    """    
    def __init__(self, local_save_dir:str="outputs", local_save:bool=False, show_debug_log:bool=False, local_log_dir:str="logs", use_as_gradio:bool=False, use_as_webapi:bool=False) -> None:
        """Initialize STELLA module.

        Args:
            local_save (bool, optional): saves output locally. default: False.
            local_save_dir (str, optional): path to save output locally. default: "outputs".
            show_debug_log (bool, optional): show detailed event logs. default: False.
            local_log_dir (str, optional): path to save log file locally. default: "logs".
            use_as_gradio (bool, optional): used for gradio frontend. default: False.
            use_as_webapi (bool, optional): used for webapi. default: False.
        """     
        if show_debug_log:
            self.local_log_dir = f"{local_log_dir}/{datetime.now().strftime('%Y-%m-%d')}"
            
            os.makedirs(
                self.local_log_dir,
                exist_ok=True)
            
            logging.basicConfig(
                format="[%(asctime)s][%(levelname)s][%(name)s@%(funcName)s:%(lineno)d] -> %(message)s",
                handlers=[
                    logging.FileHandler(f"{self.local_log_dir}/{self.__class__.__name__}_{datetime.now().strftime('%H-%M-%S')}.log", mode="a"),
                    logging.StreamHandler()
                    ],
                datefmt="%Y%m%d-%H%M%S",
                level=logging.DEBUG)
            
            self.logger = logging.getLogger(__name__)
        
        self.local_save = local_save
        self.use_as_gradio = use_as_gradio
        self.use_as_webapi = use_as_webapi
            
        if self.local_save:
            self.local_save_dir = local_save_dir
            os.makedirs(self.local_save_dir, exist_ok=True)
        
        else: self.local_save_dir = None
        
        self.load_presets()
        self.load_env()

    def prompt_randomizer(self, prompt:str=None) -> str:
        """Returns randomized prompt ideas.

        Args:
            prompt (str, optional): original prompt. default: None.

        Returns:
            str: randomized prompt
        """        
        elements = [prompt]
        
        categories = ["scene", "filter", "camera", "material", "perspective", "medium", "lighting", "rendering", "artstyle", "painter"]

        elements.extend(random.choice(self.Prompt[category]) for category in categories)
        elements = [element for element in elements if element]
        combined_prompt = ", ".join(elements)
        return combined_prompt

    def load_cubes(self, cube_dir:str) -> tuple:
        """Loads cube files.

        Args:
            cube_dir (str): the folder where cubes is stored

        Returns:
            tuple: cube dict, keys and list
        """        
        cube_dict = {}
        
        for filename in os.listdir(cube_dir):
            if filename.endswith(".cube"):
                file_path = os.path.join(cube_dir, filename)
                file_key = os.path.splitext(filename)[0]
                cube_dict[file_key] = file_path
        
        return cube_dict, cube_dict.keys(), list(cube_dict.keys())

    def list_presets(self, preset_type:str=None, export_as_list:bool=False) -> list:
        """Lists needed presets or export them as lists.

        Args:
            preset_type (str, optional): preset name. default: None.
            export_as_list (bool, optional): export as a list. default: False.

        Returns:
            list: exported list or a list of models
        """        
        preset_type = preset_type.lower() if preset_type else None
        presets = {
            "models": (self.Model, self.ModelList),
            "atelier": (self.Atelier, self.AtelierList),
            "v1": (self.V1, self.V1List),
            "v2": (self.V2, self.V2List),
            "v4": (self.V4, self.V4List),
            "anime": (self.Anime, self.AnimeList),
            "size": (self.Size, self.SizeList),
            "remix": (self.Remix, self.RemixList),
            "controlnet": (self.Controlnet, self.ControlnetList),
            "variate": (self.Variate, self.VariateList),
            "lora": (self.Lora, self.LoraList),
            "cube": (self.Cube, self.CubeList),
            "prompt": (self.Prompt, self.PromptList)
        }

        if preset_type in presets:
            return presets[preset_type][0] if export_as_list else print(presets[preset_type][1])
        
        else:
            print('''Invalid preset type. Available types are: models, v1, v2, anime, size,
                  remix, controlnet, variate, lora, cube, prompt.''')
            return None

    def load_env(self) -> None:
        """Loads environment variables file (.env)
        """        
        dotenv.load_dotenv()
        env_vars = ["SVC_URL", "SVC_KEY", "ARC_URL"]
        for var in env_vars: setattr(self, var.lower(), os.getenv(var))
        return None
    
    def load_presets(self) -> None:
        """Loads/updates all the required presets
        """        
        self.V1, self.V1Keys, self.V1List = self.load_preset("presets/V1.json")
        self.V2, self.V2Keys, self.V2List = self.load_preset("presets/V2.json")
        self.V4, self.V4Keys, self.V4List = self.load_preset("presets/V4.json")
        self.V4Control, self.V4ControlKeys, self.V4ControlList = self.load_preset("presets/Remix2.json")
        self.Anime, self.AnimeKeys, self.AnimeList = self.load_preset("presets/Anime.json")
        self.Model, self.ModelKeys, self.ModelList = self.load_preset("presets/Model.json")
        self.Atelier, self.AtelierKeys, self.AtelierList = self.load_preset("presets/Model2.json")
        self.Size, self.SizeKeys, self.SizeList = self.load_preset("presets/Size.json")
        self.Remix, self.RemixKeys, self.RemixList = self.load_preset("presets/Remix.json")
        self.Controlnet, self.ControlnetKeys, self.ControlnetList = self.load_preset("presets/Controlnet.json")
        self.Variate, self.VariateKeys, self.VariateList = self.load_preset("presets/Variate.json")
        self.Arc, self.ArcKeys, self.ArcList = self.load_preset("presets/Arc.json")
        self.Error, self.ErrorKeys, self.ErrorList = self.load_preset("presets/Error.json")
        self.Lora, self.LoraKeys, self.LoraList = self.load_preset("presets/Lora.json")
        self.Prompt, self.PromptKeys, self.PromptList = self.load_preset("presets/Prompt.json")
        self.Feature, self.FeatureKeys, self.FeatureList = self.load_preset("presets/Feature.json")
        self.Cube, self.CubeKeys, self.CubeList = self.load_cubes("./cubes")
        return None
    
    def image_lut_processor(self, image:str, cube:str) -> str:
        """Applies 3D LUT effect on an image.

        Args:
            image (str): source image file
            cube (str): 3D LUT cube file

        Returns:
            any: output image
        """        
        cubefile = load_cube_file(self.Cube[cube])
        imagefile = Image.open(image)
        result = imagefile.filter(cubefile)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as output:
            result.save(output, format="PNG")

        return output.name

    def image_generator_atelier(self, prompt:str, model_name:str="Turbo", image_size:str="Square (1:1)", number_of_images:int=1, guide_image:str=None, guide_type:str=None, denoise_strength:float=0.95, style_v4:str=None, style_preset:str=None, style_name:str=None) -> list:
        """Powerful workflow for superb quality image generation

        Args:
            prompt (str): image prompt
            negative_prompt (str): negative prompt
            image_size (str): selected image aspect ratio. default: 1:1
            face_consistency (float): accuracy of the input image. default: 1.2
            guide_image (str): input guidance image
            guide_type (str): type of guidance image
            denoise_strength (float): strength of the denoisig. default: 0.95      
            style_v4 (int): selected atelier style. default: None
            style_preset (str): selected style preset. default: None
            style_name (str): selected style name. default: None

        Returns:
            any: generated images
        """            
        if style_preset is not None and style_name is not None:
            prompt, _ = self.prompt_template(prompt, "", style_preset, style_name)
        
        if style_v4 is not None or style_v4 == "None":
            positive = self.V4[style_v4]["prompt"]
            prompt = positive.replace("{prompt}", prompt)
        
        model_name = self.Atelier[model_name]
        image_size = self.Size[image_size]
        
        url = f"{self.svc_url}/generations"
        header = {"bearer": self.svc_key}
        
        body = {
            "prompt": (None, str(prompt)),
            "style_id": (None, str(model_name)),
            "aspect_ratio": (None, str(image_size)),
            "variation": (None, "txt2img")
        }

        if guide_image is not None and guide_type is not None and guide_type.lower() != 'none':
            guide_array = BytesIO()
            Image.open(guide_image).save(guide_array, format="PNG")
            guide_array.seek(0)
            
            number_of_images = 1
            
            if guide_type.lower() == "base":
                base = {
                    "variation": (None, "img2img"),
                    "denoising_strength": (None, str(denoise_strength)),
                    "image": ("style.png", guide_array, "image/png"),
                }
                
                body.update(base)
            
            elif guide_type.lower() == "controlnet":
                cnet = {
                    "variation": (None, "txt2img"),
                    "control_1_type": (None, "depth"),
                    "control_1_image": ("style.png", guide_array, "image/png"),
                }
                
                body.update(cnet)
                
        return self.service_request(url, header, body, multiplier=number_of_images)
        

    def dual_consistency(self, image_face:str, image_style:str, prompt:str, negative_prompt:str, image_size:str="Square (1:1)", face_consistency:float=1.2, style_strength:float=0.7, image_seed:int=0, style_preset:str=None, style_name:str=None) -> str:
        """Consistant image generation with instantid and style

        Args:
            image_face (str): input face image
            image_style (str): input style image
            prompt (str): image prompt
            negative_prompt (str): negative prompt
            image_size (str): selected image aspect ratio. default: 1:1
            face_consistency (float): accuracy of the input image. default: 1.2
            style_strength (float): strength of the style image. default: 0.7        
            image_seed (int): specified image seed. default: 0
            style_preset (str): selected style preset. default: None
            style_name (str): selected style name. default: None

        Returns:
            any: generated images
        """    
        if image_face is not None:
            face_array = BytesIO()
            Image.open(image_face).save(face_array, format="PNG")
            face_array.seek(0)    

        if image_style is not None:
            style_array = BytesIO()
            Image.open(image_style).save(style_array, format="PNG")
            style_array.seek(0)
        
        if style_preset is not None and style_name is not None:
            prompt, negative_prompt = self.prompt_template(prompt, "", style_preset, style_name)
        
        if image_seed == 0: image_seed = None
        
        image_size = self.Size[image_size]
        
        url = f"{self.svc_url}/generations/consistent"
        header = {"bearer": self.svc_key}
        
        body = {
            "prompt": (None, str(prompt)),
            "negative_prompt": (None, str(negative_prompt)),
            "aspect_ratio": (None, str(image_size)),
            "identitynet_strength": (None, str(face_consistency)),
            "style_strength": (None, str(style_strength)),
            "seed": (None, image_seed),
            "style_id": (None, "3"),
            "steps": (None, "5"),
            "mode": (None, "fidelity"),
            "cfg": (None, "1.2"),
            "high_res_results": (None, "1"),
            "priority": (None, "1")
        }
        
        if image_face is not None:
            body["face_image"] = ("face.png", face_array, "image/png")

        if image_style is not None:
            body["style_image"] = ("style.png", style_array, "image/png")

        return self.service_request(url, header, body)

    def arc_face_restore(self, image:str) -> str:
        """Uses ARC to restore faces.

        Args:
            image (str): input image

        Returns:
            any: restored image
        """        
        with Image.open(image) as image:
            width, height = image.size
            
            if width > 1920 or height > 1800: 
                aspect_ratio = width / height
                
                if aspect_ratio > 1920 / 1800:
                    new_width = 1920
                    new_height = int(new_width / aspect_ratio)
                
                else:
                    new_height = 1800
                    new_width = int(new_height * aspect_ratio)
    
                image = image.resize((new_width, new_height), Image.ANTIALIAS)

            byte_array = BytesIO()
            image.save(byte_array, format="PNG")
                    
        arc_array = BytesIO(base64.b64decode(self.Arc["arc"]["data"]))
    
        url = f"{self.arc_url}"
        header = {}
        data = {"model_seltct": "1"}
        files= [
            ("file", ("file.png", byte_array, "image/png")),
            ("file2", ("file2.jpg", arc_array, "image/jpeg"))
        ]
        
        return self.service_request(url, header, files=files, data=data, arc=True)
    
    def face_identity(self, image:str, prompt:str, negative_prompt:str, image_size:str="Square (1:1)", face_consistency:float=1.0, image_seed:int=0, style_preset:str=None, style_name:str=None) -> str:
        """Consistant image generation with instantid.

        Args:
            image (str): input image
            prompt (str): image prompt
            negative_prompt (str): negative prompt
            image_size (str): selected image aspect ratio. default: 1:1
            face_consistency (float): accuracy of the input image. default: 1.0, min: 0.0, max: 1.0
            image_seed (int): specified image seed. default: 0
            style_preset (str): selected style preset. default: None
            style_name (str): selected style name. default: None

        Returns:
            any: generated images
        """    
        if image is not None:
            byte_array = BytesIO()
            Image.open(image).save(byte_array, format="PNG")
                    
        if style_preset is not None and style_name is not None:
            prompt, negative_prompt = self.prompt_template(prompt, "", style_preset, style_name)
        
        if image_seed == 0: image_seed = None
        
        image_size = self.Size[image_size]
        
        url = f"{self.svc_url}/generations/consistent"
        header = {"bearer": self.svc_key}
        
        body = {
            "prompt": (None, str(prompt)),
            "negative_prompt": (None, str(negative_prompt)),
            "aspect_ratio": (None, str(image_size)),
            "identitynet_strength": (None, str(face_consistency)),
            "seed": (None, image_seed),
            "model_version": (None, "1"),
            "image_adapter_strength": (None, "0.8"),
            "style_id": (None, "2"),
            "steps": (None, "4"),
            "fast_mode": (None, "false"),
            "canny": (None, "false"),
            "depth": (None, "false"),            
            "pose": (None, "true"),
            "cfg": (None, "1.2"),
            "high_res_results": (None, "1"),
            "priority": (None, "1")
        }
        
        if image is not None:
            body["image"] = ("input.png", byte_array, "image/png")
        
        return self.service_request(url, header, body)
    
    def realtime_canvas(self, image:str, prompt:str, lora_style:str="None", creativity_strength:float=0.875, image_seed:int=0, style_preset:str=None, style_name:str=None) -> str:
        """Instant drawing canvas.

        Args:
            image (str): composite input image
            prompt (str): image prompt
            lora_style (str): selected lora type. default: "None"
            creativity_strength (float): creativity strength. default: 0.875, min: 0.0, max: 1.0
            image_seed (int): specified image seed. default: 0
            style_preset (str): selected style preset. default: None
            style_name (str): selected style name. default: None

        Returns:
            any: generated image
        """ 
        if self.use_as_gradio: image = image["composite"]
        else: image = Image.open(image)
        
        byte_array = BytesIO()
        image.save(byte_array, format="PNG")
                
        if style_preset is not None and style_name is not None:
            prompt, _ = self.prompt_template(prompt, "", style_preset, style_name)
        
        if image_seed == 0: image_seed = None
        
        lora_style = self.Lora[lora_style]
        
        url = f"{self.svc_url}/edits/remix/turbo"
        header = {"bearer": self.svc_key}
        
        body = {
            "image": ("input.png", byte_array, "image/png"),
            "prompt": (None, str(prompt)),
            "seed": (None, image_seed),
            "lora_style": (None, str(lora_style)),
            "strength": (None, str(creativity_strength)),
            "style_id": (None, "1")
        }

        return self.service_request(url, header, body)
    
    def realtime_generator(self, prompt:str, number_of_images:int=1, lora_style:str="None", image_seed:int=0, style_preset:str=None, style_name:str=None) -> list:
        """Instant image generation.

        Args:
            prompt (str): image prompt
            number_of_images (int): number of generated images. default: 1
            lora_style (str): selected lora type. default: "None"
            image_seed (int): specified image seed. default: 0
            style_preset (str): selected style preset. default: None
            style_name (str): selected style name. default: None

        Returns:
            list: generated images
        """
        if style_preset is not None and style_name is not None:
            prompt, _ = self.prompt_template(prompt, "", style_preset, style_name)
        
        if image_seed == 0: image_seed = None
        
        lora_style = self.Lora[lora_style]
        
        url = f"{self.svc_url}/generations/turbo"
        header = {"bearer": self.svc_key}
        
        body = {
            "prompt": (None, str(prompt)),
            "seed": (None, image_seed),
            "lora_style": (None, str(lora_style)),
            "style_id": (None, "1")
        }
        
        return self.service_request(url, header, body, multiplier=number_of_images)
    
    def image_inpainting(self, image:str, prompt:str, negative_prompt:str, inpaint_strength:float=0.5, prompt_scale:float=9.0, image_mask:str=None, style_preset:str=None, style_name:str=None) -> str:
        """Inpaint elements into an image.

        Args:
            image (str): input image
            prompt (str): image prompt
            negative_prompt (str): negative prompt
            inpaint_strength (float): strength of inpainting. default: 0.5, min: 0.0, max: 1.0
            prompt_scale (float): scale of prompt/creativity. default: 9.0
            image_mask (str): mask image filepath. default: None
            style_preset (str): selected style preset. default: None
            style_name (str): selected style name. default: None

        Returns:
            any: output image
        """        
        if self.use_as_gradio:
            src_img = image["background"]
            mask_layer = image["layers"][0]
            mask_np_arr = numpy.array(mask_layer)
            mask_np_img = numpy.where(mask_np_arr[:, :, 3] == 0, 0, 255).astype(numpy.uint8)
            mask_img = Image.fromarray(mask_np_img)
            
        else:
            src_img = Image.open(image)
            mask_img = Image.open(image_mask)

        source_image = BytesIO()
        src_img.save(source_image, format="PNG")
        source_image.seek(0)

        mask_image = BytesIO()
        mask_img.save(mask_image, format="PNG")
        mask_image.seek(0)
        
        if style_preset is not None and style_name is not None:
            prompt, negative_prompt = self.prompt_template(prompt, "", style_preset, style_name)
    
        url = f"{self.svc_url}/edits/inpaint"
        header = {"bearer": self.svc_key}
        
        body = {
            "image": ("image.png", source_image, "image/png"),
            "mask": ("mask.png", mask_image, "image/png"),
            "prompt": (None, str(prompt)),
            "neg_prompt": (None, str(negative_prompt)),
            "inpaint_strength": (None, str(inpaint_strength)),
            "cfg": (None, str(prompt_scale)),
            "priority": (None, "1")
        }

        return self.service_request(url, header, body)
    
    def image_eraser(self, image:str, image_mask:str=None, prompt_scale:float=9.0) -> str:
        """Erase specific elements from an image.

        Args:
            image (dict): background and mask (white on black) images.
            mask (Image, optional): mask (white on black) image. default: None.
            prompt_scale (float): scale of prompt/creativity. default: 9.0

        Returns:
            any: erased image
        """
        if self.use_as_gradio:
            src_img = image["background"]
            mask_layer = image["layers"][0]
            mask_np_arr = numpy.array(mask_layer)
            mask_np_img = numpy.where(mask_np_arr[:, :, 3] == 0, 0, 255).astype(numpy.uint8)
            mask_img = Image.fromarray(mask_np_img)

        else:
            src_img = Image.open(image)
            mask_img = Image.open(image_mask)

        source_image = BytesIO()
        src_img.save(source_image, format="PNG")
        source_image.seek(0)

        mask_image = BytesIO()
        mask_img.save(mask_image, format="PNG")
        mask_image.seek(0)
    
        url = f"{self.svc_url}/edits/remove"
        header = {"bearer": self.svc_key}
        
        body = {
            "image": ("image.png", source_image, "image/png"),
            "mask": ("mask.png", mask_image, "image/png"),
            "cfg": (None, str(prompt_scale)),
            "model_version": (None, "1"),
            "priority": (None, "1")
        }

        return self.service_request(url, header, body)

    def creative_upscaler(self, image:str, prompt:str, negative_prompt:str, creativity_strength:float=0.5, resemblance_strength:float=0.8, hdr_strength:float=0.5, style_preset:str=None, style_name:str=None) -> str:
        """Generative image upscaler.

        Args:
            image (str): input image
            prompt (str): image prompt
            negative_prompt (str): negative prompt
            creativity_strength (float): strength of creativeness. default: 0.5, min: 0.2, max: 1.0
            resemblance_strength (float): strength of resemblance. default: 0.8, min: 0.0, max: 1.0
            hdr_strength (float): strength of hdr effect. default: 0.5, min: 0.0, max: 1.0
            style_preset (str): selected style preset. default: None
            style_name (str): selected style name. default: None

        Returns:
            any: upscaled image
        """        
        with Image.open(image) as image:
            byte_array = BytesIO()
            image.save(byte_array, format="PNG")
            
        if style_preset is not None and style_name is not None:
            prompt, negative_prompt = self.prompt_template(prompt, "", style_preset, style_name)
        
        url = f"{self.svc_url}/enhance"
        header = {"bearer": self.svc_key}
        
        body = {
            "image": ("input.png", byte_array, "image/png"),
            "prompt": (None, str(prompt)),
            "hdr": (None, str(hdr_strength)),
            "creativity": (None, str(creativity_strength)),
            "resemblance": (None, str(resemblance_strength)),
            "negativePrompt": (None, str(negative_prompt)),
            "negative_prompt": (None, str(negative_prompt)),
            "model_version": (None, "1"),
            "style_id": (None, "6")
        }

        return self.service_request(url, header, body)

    def image_variation(self, image:str, prompt:str, negative_prompt:str, model_name:str="V3", variate_strength:float=0.85, prompt_scale:float=9.0, image_seed:int=0, style_preset:str=None, style_name:str=None) -> str:
        """Make variations of an image.

        Args:
            image (str): input image
            prompt (str): image prompt
            negative_prompt (str): negative prompt
            model_name (str): selected model name. default: v3
            variate_strength (float): strength of variation. default: 0.85, min:0.0, max:1.0
            prompt_scale (float): scale of prompt/creativity. default: 9.0
            image_seed (int): specified image seed. default: 0
            style_preset (str): selected style preset. default: None
            style_name (str): selected style name. default: None

        Returns:
            any: variation of an image
        """        
        byte_array = BytesIO()
        Image.open(image).save(byte_array, format="PNG")
        
        if style_preset is not None and style_name is not None:
            prompt, negative_prompt = self.prompt_template(prompt, "", style_preset, style_name)
            
        model_name = self.Variate[model_name]
        if image_seed == 0: image_seed = None
        
        url = f"{self.svc_url}/generations/variations"
        header = {"bearer": self.svc_key}
        
        body = {
            "image": ("input.png", byte_array, "image/png"),
            "prompt": (None, str(prompt)),
            "style_id": (None, str(model_name)),
            "strength": (None, str(variate_strength)),
            "cfg": (None, str(prompt_scale)),
            "negative_prompt": (None, str(negative_prompt)),
            "seed": (None, image_seed),
            "model_version": (None, "1"),
            "prompt_processed": (None, "0"),
            "priority": (None, "1")
        }

        return self.service_request(url, header, body)

    def image_controlnet(self, image:str, prompt:str, negative_prompt:str, model_name:str="Toon", control_type:str="Scribble", control_strength:int=70, prompt_scale:float=9.0, image_seed:int=0, style_preset:str=None, style_name:str=None) -> str:
        """Controls an image into a different subject.

        Args:
            image (str): input image
            prompt (str): image prompt
            negative_prompt (str): negative prompt
            model_name (str): selected model name. default: toon
            control_type (str): type of controlnet. default: scribble
            control_strength (int): strength of controlnet. default: 70, min: 0, max: 100
            prompt_scale (float): scale of prompt/creativity. default: 9.0
            image_seed (int): specified image seed. default: 0
            style_preset (str): selected style preset. default: None
            style_name (str): selected style name. default: None

        Returns:
            any: remixed image
        """        
        with Image.open(image) as image:
            byte_array = BytesIO()
            image.save(byte_array, format="PNG")
            
        if style_preset is not None and style_name is not None:
            prompt, negative_prompt = self.prompt_template(prompt, "", style_preset, style_name)
        
        model_name = self.Remix[model_name]
        control_type = self.Controlnet[control_type]
        if image_seed == 0: image_seed = None
        
        url = f"{self.svc_url}/edits/remix"
        header = {"bearer": self.svc_key}
        
        body = {
            "image": ("input.png", byte_array, "image/png"),
            "model_version": (None, "1"),
            "prompt": (None, str(prompt)),
            "cfg": (None, str(prompt_scale)),
            "style_id": (None, str(model_name)),
            "control": (None, str(control_type)),
            "strength": (None, str(control_strength)),
            "negative_prompt": (None, str(negative_prompt)),
            "seed": (None, image_seed),
            "priority": (None, "1")
        }

        return self.service_request(url, header, body)

    def image_upscaler(self, image:str) -> str:
        """Upscales an image.

        Args:
            image (str): input image

        Returns:
            any: upscaled image
        """        
        with Image.open(image) as image:
            byte_array = BytesIO()
            image.save(byte_array, format="PNG")
            
        url = f"{self.svc_url}/upscale"
        header = {"bearer": self.svc_key}
        
        body = {
            "image": ("input.png", byte_array, "image/png"),
            "model_version": (None, "1")
        }

        return self.service_request(url, header, body)

    def background_remover(self, image:str) -> str:
        """Removes background from an image.

        Args:
            image (str): input image

        Returns:
            any: processed image
        """        
        with Image.open(image) as image:
            byte_array = BytesIO()
            image.save(byte_array, format="PNG")
            
        url = f"{self.svc_url}/background/remover"
        header = {"bearer": self.svc_key}
        
        body = {
            "image": ("input.png", byte_array, "image/png"),
            "model_version": (None, "1")
        }

        return self.service_request(url, header, body)
    
    def prompt_generator(self, image:str) -> str:
        """Reads input image as a prompt.

        Args:
            image (str): input image

        Returns:
            str: text prompt
        """        
        with Image.open(image) as image:
            byte_array = BytesIO()
            image.save(byte_array, format="PNG")
            
        url = f"{self.svc_url}/generations/image"
        header = {"bearer": self.svc_key}
        
        body = {
            "image": ("input.png", byte_array, "image/png"),
            "model_version": (None, "1")
        }

        return self.service_request(url, header, body) #.split(",", 1)[0]
    
    def image_generator(self, prompt:str, negative_prompt:str, model_name:str="Turbo", image_size:str="Square (1:1)", number_of_images:int=1, prompt_scale:float=9.0, image_seed:int=0, style_preset:str=None, style_name:str=None) -> list:
        """High quality image generator.

        Args:
            prompt (str): image prompt
            negative_prompt (str): negative prompt
            model_name (str): selected model name. default: turbo
            image_size (str): selected image aspect ratio. default: 1:1
            number_of_images (int): number of generated images. default: 1
            prompt_scale (float): scale of prompt/creativity. default: 9.0
            image_seed (int): specified image seed. default: 0
            style_preset (str): selected style preset. default: None
            style_name (str): selected style name. default: None

        Returns:
            list: generated images
        """
        if style_preset is not None and style_name is not None:
            prompt, negative_prompt = self.prompt_template(prompt, "", style_preset, style_name)
        
        if image_seed == 0: image_seed = None
        
        model_name = self.Model[model_name]
        image_size = self.Size[image_size]
        
        url = f"{self.svc_url}/generations"
        header = {"bearer": self.svc_key}
        
        body = {
            "model_version": (None, "1"),
            "prompt": (None, str(prompt)),
            "style_id": (None, str(model_name)),
            "negative_prompt": (None, str(negative_prompt)),
            "aspect_ratio": (None, str(image_size)),
            "seed": (None, image_seed),
            "cfg": (None, str(prompt_scale)),
            "high_res_results": (None, "1"),
            "priority": (None, "1")
        }
        
        return self.service_request(url, header, body, multiplier=number_of_images)
        # generated_images = []
        
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(self.service_request, url, header, body) for _ in range(number_of_images)]
            
        #     for future in concurrent.futures.as_completed(futures):
        #         result = future.result()
        #         if result is not None:
        #             generated_images.append(result)
    
        # return generated_images
    
    def image_enhancer(self, image:str, sharpness:float=1.5, brightness:float=1.025, color:float=1.05, contrast:float=1.025) -> str:
        """Improves image quality on various levels.

        Args:
            image (str): source image
            sharpness (float, optional): image sharpness. default: 1.5.
            brightness (float, optional): image brightness. default: 1.025.
            color (float, optional): image saturation. default: 1.05.
            contrast (float, optional): image contrast. default: 1.025.

        Returns:
            Image: enhanced image
        """
        
        with Image.open(image) as original_image:
            enhanced_image = ImageEnhance.Contrast(
                ImageEnhance.Color(
                    ImageEnhance.Brightness(
                        ImageEnhance.Sharpness(
                            original_image
                        ).enhance(sharpness)
                    ).enhance(brightness)
                ).enhance(color)
            ).enhance(contrast)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"{self.__class__.__name__}.png") as output:
                enhanced_image.save(output.name)

            return output.name
    
    def prompt_template(self, prompt:str, negative_prompt:str, style_preset:str, style_name:str) -> tuple[str, str]:
        """Process user prompt with their choosen style library and name.

        Args:
            prompt (str): prompt
            negative_prompt (str): negative prompt
            style_preset (str): selected style preset.
            style_name (str): selected style name.

        Returns:
            tuple[str, str]: processed prompt and negative prompt
        """        
        if style_preset == "None" and style_name == "None":
            return prompt, negative_prompt
        
        else:
            style_dict = {"V1": self.V1, "V2": self.V2, "Anime": self.Anime}
            positive = style_dict[style_preset][style_name]["prompt"]
            negative = style_dict[style_preset][style_name]["negative_prompt"]
            
            return positive.replace("{prompt}", prompt), negative.replace("{negative_prompt}", negative_prompt)
    
    def load_preset(self, preset:str) -> tuple:
        """Loads json files and turn them into a library

        Args:
            preset (str): path to json file

        Returns:
            tuple: the library, the keys and the list
        """        
        preset = json.load(open(preset, encoding="utf-8"))
        return preset, preset.keys(), list(preset.keys())
    
    def save_temp_file(self, content, suffix) -> str:
        """Helper method to set up the temp directory, save content to a temporary file, and log the action.

        Args:
            content (bytes): Content to be written to the file.
            suffix (str): File suffix (e.g., .png, .txt).

        Returns:
            str: The path to the saved file.
        """
        # Set up the temp_dir only once
        if self.local_save:
            temp_dir = str(self.local_save_dir)
            temp_dir = os.path.join(temp_dir, datetime.now().strftime("%Y-%m-%d"))
            os.makedirs(temp_dir, exist_ok=True)
        
        else: temp_dir = None

        with tempfile.NamedTemporaryFile(delete=False, prefix=f"{self.__class__.__name__}_", suffix=suffix, dir=temp_dir) as temp_file:
            temp_file.write(content)
        
        self.logger.debug(f"Saved output: {temp_file.name}")
        return temp_file.name

    def service_request(self, url:str, header:dict, files:dict, data:dict=None, tx_timeout:int=90, rx_timeout:int=90, delay:float=0.5, multiplier:int=1, arc:bool=False) -> list:
        """Process inputs for each server connection concurrently.

        Args:
            url (str): service url
            header (dict): header for post request
            files (dict): data for post request
            data (dict): data for post request
            tx_timeout (int): transmit timeout in seconds. default: 90
            rx_timeout (int): receive timeout in seconds. default: 90
            delay (float): delay time in seconds. default: 0.5
            multiplier (int): number of concurrent requests. default: 1
            arc (bool): whether to use arc-specific processing. default: False

        Returns:
            list: A list of results from the requests.
        """

        def request_handler():
            try:
                time.sleep(delay)
                start_time = time.time()
                
                if arc:
                    response = requests.post(url, headers=header, data=data, files=files).json()
                    result = response["data"][0]["image_base64"].split(",")[1]
                    content = base64.b64decode(result)
                    return self.save_temp_file(content, ".png")
                
                else:
                    response = requests.post(url, headers=header, files=files, timeout=(tx_timeout, rx_timeout))
                    content_type = response.headers.get("Content-Type", "").lower()

                    if response.status_code == 200:
                        if "image" in content_type:
                            error_array = BytesIO(base64.b64decode(self.Error["error"]["data"])).read()
                            if response.content == error_array:
                                return None
                            return self.save_temp_file(response.content, ".png")
                        
                        elif "text" in content_type:
                            return self.save_temp_file(response.text.encode('utf-8'), ".txt")
                    
                    else:
                        return None
            
            except requests.exceptions.Timeout:
                self.logger.warning("Request timeout!")
                return None

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed: {e}")
                return None
            
            finally:
                end_time = time.time()
                request_time = end_time - start_time
                self.logger.warning(f"The request took in {request_time:.2f} seconds.")

        received_requests = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(request_handler) for _ in range(multiplier)]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    received_requests.append(result)
        
        return received_requests

if __name__ == "__main__":
    print("ikmalsaid's STELLA AI Studio (Version 24.0731). Copyright (C) 2024 All rights reserved.")
    print('''Currently supporting these 18 features:
- Image generation\t\t- Image creative upscaling
- Image to image\t\t- Image background remover
- Image variation\t\t- Prompt randomizer
- Image inpainting\t\t- Realtime generation
- Image eraser\t\t\t- Realtime canvas
- Image upscaling\t\t- Face consistency
- Image enhancer\t\t- Style consistency
- Image 3D LUT processor\t- Face restoration
- Image prompt generator\t- Face swapping''')