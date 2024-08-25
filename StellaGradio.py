import gradio as gr
from StellaApp import StellaApp
from StellaD3 import StellaD3

sa = StellaApp(use_as_gradio=True, local_save=True, show_debug_log=True)
d3 = StellaD3()

fit_box = 16
styles_preset = ["None", "Anime", "V1", "V2"]
image_types = ["jpg", "jpeg", "png", "webp"]

system_theme = gr.themes.Default(
    font=[gr.themes.GoogleFont("Segoe UI")],
    font_mono=[gr.themes.GoogleFont("Segoe UI")],
    primary_hue=gr.themes.colors.rose,
    secondary_hue=gr.themes.colors.rose,
    neutral_hue=gr.themes.colors.zinc,
    )

def f1_preprocess(f1_pro, f1_neg, f1_mod, f1_siz, f1_num, f1_sca, f1_sed, f1_pre, f1_sty, f1_ram):
    results = sa.image_generator(f1_pro, f1_neg, f1_mod, f1_siz, f1_num, f1_sca, f1_sed, f1_pre, f1_sty)
    for i in results: f1_ram.insert(0, i)
    return f1_ram

def f2_preprocess(f2_img, f2_pro, f2_neg, f2_mod, f2_con, f2_str, f2_sca, f2_sed, f2_pre, f2_sty, f2_ram):
    results = sa.image_controlnet(f2_img, f2_pro, f2_neg, f2_mod, f2_con, f2_str, f2_sca, f2_sed, f2_pre, f2_sty)
    for i in results: f2_ram.insert(0, i)
    return f2_ram

def f3_preprocess(f3_img, f3_pro, f3_neg, f3_mod, f3_str, f3_sca, f3_sed, f3_pre, f3_sty, f3_ram):
    results = sa.image_variation(f3_img, f3_pro, f3_neg, f3_mod, f3_str, f3_sca, f3_sed, f3_pre, f3_sty)
    for i in results: f3_ram.insert(0, i)
    return f3_ram

def f4_preprocess(f4_img, f4_ram):
    results = sa.image_upscaler(f4_img)
    for i in results: f4_ram.insert(0, i)
    return f4_ram

def f5_preprocess(f5_img, f4_ram):
    results = sa.background_remover(f5_img)
    for i in results: f4_ram.insert(0, i)
    return f4_ram

def f7_preprocess(f7_img, f7_pro, f7_neg, f7_cre, f7_rsm, f7_hdr, f7_pre, f7_sty, f7_ram):
    results = sa.creative_upscaler(f7_img, f7_pro, f7_neg, f7_cre, f7_rsm, f7_hdr, f7_pre, f7_sty)
    for i in results: f7_ram.insert(0, i)
    return f7_ram

def f8_preprocess(f8_mas, f8_sca, f8_ram):
    results = sa.image_eraser(f8_mas, f8_sca)
    for i in results: f8_ram.insert(0, i)
    return f8_ram

def f9_preprocess(f9_mas, f9_pro, f9_neg, f9_str, f9_sca, f9_pre, f9_sty, f9_ram):
    results = sa.image_inpainting(f9_mas, f9_pro, f9_neg, f9_str, f9_sca, f9_pre, f9_sty)
    for i in results: f9_ram.insert(0, i)
    return f9_ram

def f10_preprocess(f10_pro, f10_num, f10_lra, f10_sed, f10_pre, f10_sty, f10_ram):
    results = sa.realtime_generator(f10_pro, f10_num, f10_lra, f10_sed, f10_pre, f10_sty)
    for i in results: f10_ram.insert(0, i)
    return f10_ram

def f11_preprocess(f11_can, f11_pro, f11_lra, f11_str, f11_sed, f11_pre, f11_sty, f11_ram):
    results = sa.realtime_canvas(f11_can, f11_pro, f11_lra, f11_str, f11_sed, f11_pre, f11_sty)
    for i in results: f11_ram.insert(0, i)
    return f11_ram

def f12_preprocess(f12_img, f12_ram):
    results = sa.arc_face_restore(f12_img)
    for i in results: f12_ram.insert(0, i)
    return f12_ram

def f13_preprocess(f13_fce, f13_stl, f13_pro, f13_neg, f13_siz, f13_fco, f13_sst, f13_sed, f13_pre, f13_sty, f13_ram):
    results = sa.dual_consistency(f13_fce, f13_stl, f13_pro, f13_neg, f13_siz, f13_fco, f13_sst, f13_sed, f13_pre, f13_sty)
    for i in results: f13_ram.insert(0, i)
    return f13_ram

def f14_preprocess(f14_fce, f14_pro, f14_neg, f14_siz, f14_fco, f14_sed, f14_pre, f14_sty, f14_ram):
    results = sa.face_identity(f14_fce, f14_pro, f14_neg, f14_siz, f14_fco, f14_sed, f14_pre, f14_sty)
    for i in results: f14_ram.insert(0, i)
    return f14_ram

def f15_preprocess(f15_pro, f15_mod, f15_siz, f15_num, f15_gdi, f15_gdt, f15_den, f15_v4, f15_pre, f15_sty, f15_ram):
    results = sa.image_generator_atelier(f15_pro, f15_mod, f15_siz, f15_num, f15_gdi, f15_gdt, f15_den, f15_v4, f15_pre, f15_sty)
    for i in results: f15_ram.insert(0, i)
    return f15_ram

def f16_preprocess(f16_pro, f16_ram):
    results = d3.image_generator_d3(f16_pro)
    for i in results: f16_ram.insert(0, i)
    return f16_ram
        
def preset_select(preset):
    if preset == "None":
        return gr.Dropdown(choices=["None"], value="None")
    else:
        styleList_name = f"{preset}List"
        styleList = getattr(sa, styleList_name)
        return gr.Dropdown(choices=styleList, value=styleList[0])

with gr.Blocks(title="AI Studio", css="html/Style.css", analytics_enabled=False, theme=system_theme, fill_height=True) as demo:
    gr.Markdown("## <br><center>Stella AI Studio - Gradio Playground Version 24.0731")

    with gr.Tab("Image Generator"):
        with gr.Row(equal_height=False):
            with gr.Column(variant="panel", scale=1) as menu:
                gr.Markdown("## <center>Image Generator")
                gr.Markdown("<center>Basic Settings")
                f1_pro = gr.Textbox(placeholder="Prompt for image...", container=False, lines=5, max_lines=5)
                f1_neg = gr.Textbox(placeholder="Negative prompt...", container=False, lines=5, max_lines=5)
                f1_num = gr.Slider(value=4, minimum=1, maximum=8, step=1, label="Number of Images")
                
                gr.Markdown("<center>Advanced Settings")
                f1_mod = gr.Dropdown(choices=sa.ModelList, value="Turbo", container=False)
                f1_siz = gr.Dropdown(choices=sa.SizeList, value=sa.SizeList[0], container=False)
                f1_sca = gr.Slider(value=9, minimum=3, maximum=15, step=0.05, label="Prompt Scale")
                f1_sed = gr.Number(value=0, minimum=0, label="Seed (0 for Random)")
                
                gr.Markdown("<center>Style Presets")
                f1_pre = gr.Dropdown(choices=styles_preset, value="None", container=False)
                f1_sty = gr.Dropdown(choices=["None"], value="None", container=False)
                f1_sub = gr.Button("Generate", variant="stop")
                f1_pre.change(fn=preset_select, inputs=[f1_pre], outputs=[f1_sty], show_progress="hidden")

            with gr.Column(variant="panel", scale=3) as result:
                f1_res = gr.Gallery(height=945.875, object_fit="contain", container=False, show_share_button=False)
                f1_ram = gr.State([])
                f1_sub.click(
                    show_api=False,
                    scroll_to_output=True,
                    fn=f1_preprocess,
                    inputs=[f1_pro, f1_neg, f1_mod, f1_siz, f1_num, f1_sca, f1_sed, f1_pre, f1_sty, f1_ram],
                    outputs=[f1_res]
                )
    
    with gr.Tab("Image Controlnet"):
        with gr.Row(equal_height=False):
            with gr.Column(variant="panel", scale=1) as menu:
                gr.Markdown("## <center>Image Controlnet")
                gr.Markdown("<center>Basic Settings")
                f2_img = gr.Image(type="filepath", height=199, sources=["upload"], label="Upload Image")
                f2_pro = gr.Textbox(placeholder="Prompt for image...", container=False)
                f2_neg = gr.Textbox(placeholder="Negative prompt...", container=False)
               
                gr.Markdown("<center>Advanced Settings")
                f2_mod = gr.Dropdown(choices=sa.RemixList, value="Toon", container=False)
                f2_con = gr.Dropdown(choices=sa.ControlnetList, value=sa.ControlnetList[0], container=False)
                f2_str = gr.Slider(value=70, minimum=0, maximum=100, step=1, label="Controlnet Strength")
                f2_sca = gr.Slider(value=9, minimum=3, maximum=15, step=0.05, label="Prompt Scale")
                f2_sed = gr.Number(value=0, minimum=0, label="Seed (0 for Random)")
                
                gr.Markdown("<center>Style Presets")
                f2_pre = gr.Dropdown(choices=styles_preset, value="None", container=False)
                f2_sty = gr.Dropdown(choices=["None"], value="None", container=False)
                f2_sub = gr.Button("Generate", variant="stop")
                f2_pre.change(fn=preset_select, inputs=[f2_pre], outputs=[f2_sty], show_progress="hidden")
                
            with gr.Column(variant="panel", scale=3) as result:
                f2_res = gr.Gallery(height=991.94, container=False, elem_id="f2_res", show_share_button=False)
                f2_ram = gr.State([])
                f2_sub.click(
                    show_api=False,
                    scroll_to_output=True,
                    fn=f2_preprocess,
                    inputs=[f2_img, f2_pro, f2_neg, f2_mod, f2_con, f2_str, f2_sca, f2_sed, f2_pre, f2_sty, f2_ram],
                    outputs=[f2_res]
                )    
    
    with gr.Tab("Image Variation"):
        with gr.Row(equal_height=False):
            with gr.Column(variant="panel", scale=1) as menu:
                gr.Markdown("## <center>Image Variation")
                gr.Markdown("<center>Basic Settings")
                f3_img = gr.Image(type="filepath", height=199, sources=["upload"], label="Upload Image")
                f3_pro = gr.Textbox(placeholder="Prompt for image...", container=False)
                f3_neg = gr.Textbox(placeholder="Negative prompt...", container=False)
               
                gr.Markdown("<center>Advanced Settings")
                f3_mod = gr.Dropdown(choices=sa.VariateList, value="V3", container=False)
                f3_str = gr.Slider(value=0.85, minimum=0.0, maximum=1.0, step=0.05, label="Variate Strength")
                f3_sca = gr.Slider(value=9, minimum=3, maximum=15, step=0.05, label="Prompt Scale")
                f3_sed = gr.Number(value=0, minimum=0, label="Seed (0 for Random)")

                
                gr.Markdown("<center>Style Presets")
                f3_pre = gr.Dropdown(choices=styles_preset, value="None", container=False)
                f3_sty = gr.Dropdown(choices=["None"], value="None", container=False)
                f3_sub = gr.Button("Generate", variant="stop")
                f3_pre.change(fn=preset_select, inputs=[f3_pre], outputs=[f3_sty], show_progress="hidden")

            with gr.Column(variant="panel", scale=3) as result:
                f3_res = gr.Gallery(height=936.938, container=False, elem_id="f3_res", show_share_button=False)
                f3_ram = gr.State([])
                f3_sub.click(
                    show_api=False,
                    scroll_to_output=True,
                    fn=f3_preprocess,
                    inputs=[f3_img, f3_pro, f3_neg, f3_mod, f3_str, f3_sca, f3_sed, f3_pre, f3_sty, f3_ram],
                    outputs=[f3_res]
                )    
        
    with gr.Tab("Image Toolkit"):
        with gr.Row(equal_height=False):
            with gr.Column(variant="panel", scale=1) as menu:
                gr.Markdown("## <center>Image 2 + 1")
                gr.Markdown("<center>4X Upscaler")
                f4_img = gr.Image(type="filepath", height=199, sources=["upload"], label="Upload Image")
                f4_sub = gr.Button("Upscale Image", variant="stop")

                gr.Markdown("<center>Background Remover")
                f5_img = gr.Image(type="filepath", height=199, sources=["upload"], label="Upload Image")
                f5_sub = gr.Button("Remove Background", variant="stop")
                
                gr.Markdown("<center>Image Prompter")
                f6_img = gr.Image(type="filepath", height=199, sources=["upload"], label="Upload Image")
                f6_sub = gr.Button("Describe Image", variant="stop")

            with gr.Column(variant="panel", scale=3) as result:
                f4_res = gr.Gallery(height=818.406, container=False, elem_id="f4_res", show_share_button=False)
                f4_ram = gr.State([])
                f6_res = gr.Textbox(placeholder="Upload and describe an image to get a prompt...", container=False, lines=5, max_lines=5)
                
                f4_sub.click(
                    show_api=False,
                    scroll_to_output=True,
                    fn=f4_preprocess,
                    inputs=[f4_img, f4_ram],
                    outputs=[f4_res] 
                )
                
                f5_sub.click(
                    show_api=False,
                    scroll_to_output=True,
                    fn=f5_preprocess,
                    inputs=[f5_img, f4_ram],
                    outputs=[f4_res] 
                )
                
                f6_sub.click(
                    show_api=False,
                    scroll_to_output=True,
                    fn=sa.prompt_generator,
                    inputs=[f6_img],
                    outputs=[f6_res] 
                ) 
        
    with gr.Tab("Creative Upscale"):
        with gr.Row(equal_height=False):
            with gr.Column(variant="panel", scale=1) as menu:
                gr.Markdown("## <center>Creative Upscale")
                gr.Markdown("<center>Basic Settings")
                f7_img = gr.Image(type="filepath", height=279, sources=["upload"], label="Upload Image")
                f7_pro = gr.Textbox(placeholder="Prompt for image...", container=False)
                f7_neg = gr.Textbox(placeholder="Negative prompt...", container=False)
               
                gr.Markdown("<center>Advanced Settings")
                f7_cre = gr.Slider(value=0.5, minimum=0.2, maximum=1.0, step=0.05, label="Creativity Strength")
                f7_rsm = gr.Slider(value=1.0, minimum=0.0, maximum=1.0, step=0.05, label="Resemblance Strength")
                f7_hdr = gr.Slider(value=0.5, minimum=0.0, maximum=1.0, step=0.05, label="HDR Strength")
                
                gr.Markdown("<center>Style Presets")
                f7_pre = gr.Dropdown(choices=styles_preset, value="None", container=False)
                f7_sty = gr.Dropdown(choices=["None"], value="None", container=False)
                f7_sub = gr.Button("Generate", variant="stop")
                f7_pre.change(fn=preset_select, inputs=[f7_pre], outputs=[f7_sty], show_progress="hidden")

            with gr.Column(variant="panel", scale=3) as result:
                f7_res = gr.Gallery(container=False, height=939.938, elem_id="f7_res", show_share_button=False)
                f7_ram = gr.State([])
                f7_sub.click(
                    show_api=False,
                    scroll_to_output=True,
                    fn=f7_preprocess,
                    inputs=[f7_img, f7_pro, f7_neg, f7_cre, f7_rsm, f7_hdr, f7_pre, f7_sty, f7_ram],
                    outputs=[f7_res]
                )            
        
    with gr.Tab("Object Eraser"):
        with gr.Row(equal_height=True):
            with gr.Column(variant="panel", scale=1) as menu:
                gr.Markdown("## <center>Object Eraser")
                gr.Markdown("<center>Advanced Settings")
                f8_sca = gr.Slider(value=9, minimum=3, maximum=15, step=0.05, label="Prompt Scale")
                f8_sub = gr.Button("Erase Object", variant="stop")
             
            with gr.Column(variant="panel", scale=2) as input:
                f8_mas = gr.ImageMask(
                    type="pil",
                    layers=False,
                    container=False,
                    transforms=[],
                    sources=["upload"],
                    brush=gr.Brush(default_size=24, colors=["#000000"], color_mode="fixed"),
                    eraser=gr.Eraser(default_size=24),
                    canvas_size=(1024,1024)
                )
            with gr.Column(variant="panel", scale=2) as result:
                f8_res = gr.Gallery(container=False, elem_id="f11_res", show_share_button=False)
                f8_ram = gr.State([])
                f8_sub.click(
                    show_api=False,
                    scroll_to_output=True,
                    fn=f8_preprocess,
                    inputs=[f8_mas, f8_sca, f8_ram],
                    outputs=[f8_res]
                ) 
                           
    with gr.Tab("Image Inpainting"):
        with gr.Row(equal_height=True):
            with gr.Column(variant="panel", scale=1) as menu:
                gr.Markdown("## <center>Image Inpainting")
                gr.Markdown("<center>Basic Settings")
                f9_pro = gr.Textbox(placeholder="Prompt for image...", container=False, lines=2, max_lines=2)
                f9_neg = gr.Textbox(placeholder="Negative prompt...", container=False, lines=1, max_lines=1)
               
                gr.Markdown("<center>Advanced Settings")
                f9_str = gr.Slider(value=0.5, minimum=0.0, maximum=1.0, step=0.05, label="Inpaint Strength")
                f9_sca = gr.Slider(value=9, minimum=3, maximum=15, step=0.05, label="Prompt Scale")

                gr.Markdown("<center>Style Presets")
                f9_pre = gr.Dropdown(choices=styles_preset, value="None", container=False)
                f9_sty = gr.Dropdown(choices=["None"], value="None", container=False)
                f9_sub = gr.Button("Inpaint Image", variant="stop")
                f9_pre.change(fn=preset_select, inputs=[f9_pre], outputs=[f9_sty], show_progress="hidden")
        
            with gr.Column(variant="panel", scale=2) as input:
                f9_mas = gr.ImageMask(
                    type="pil",
                    layers=False,
                    container=False,
                    transforms=[],
                    sources=["upload"],
                    brush=gr.Brush(default_size=24, colors=["#000000"], color_mode="fixed"),
                    eraser=gr.Eraser(default_size=24),
                    canvas_size=(1024,1024)
                )
            with gr.Column(variant="panel", scale=2) as result:
                f9_res = gr.Gallery(container=False, elem_id="f11_res", show_share_button=False)
                f9_ram = gr.State([])
                f9_sub.click(
                    show_api=False,
                    scroll_to_output=True,
                    fn=f9_preprocess,
                    inputs=[f9_mas, f9_pro, f9_neg, f9_str, f9_sca, f9_pre, f9_sty, f9_ram],
                    outputs=[f9_res]
                )

    with gr.Tab("RT Generator"):
        with gr.Row(equal_height=False):
            with gr.Column(variant="panel", scale=1) as menu:
                gr.Markdown("## <center>RT Image Generator")
                gr.Markdown("<center>Basic Settings")
                f10_pro = gr.Textbox(placeholder="Prompt for image...", container=False, lines=5, max_lines=5)
                f10_num = gr.Slider(value=4, minimum=1, maximum=8, step=1, label="Number of Images")
                
                gr.Markdown("<center>Advanced Settings")
                f10_lra = gr.Dropdown(choices=sa.LoraList, value="None", container=False)
                f10_sed = gr.Number(value=0, minimum=0, label="Seed (0 for Random)")
                
                gr.Markdown("<center>Style Presets")
                f10_pre = gr.Dropdown(choices=styles_preset, value="None", container=False)
                f10_sty = gr.Dropdown(choices=["None"], value="None", container=False)
                f10_sub = gr.Button("Generate", variant="stop")
                f10_pre.change(fn=preset_select, inputs=[f10_pre], outputs=[f10_sty], show_progress="hidden")

            with gr.Column(variant="panel", scale=3) as result:
                f10_res = gr.Gallery(height=945.875, object_fit="contain", container=False, show_share_button=False, columns=4)
                f10_ram = gr.State([])
                f10_sub.click(
                    show_api=False,
                    scroll_to_output=True,
                    fn=f10_preprocess,
                    inputs=[f10_pro, f10_num, f10_lra, f10_sed, f10_pre, f10_sty, f10_ram],
                    outputs=[f10_res]
                )

    with gr.Tab("RT Canvas"):
        with gr.Row(equal_height=False):
            with gr.Column(variant="panel", scale=1) as menu:
                gr.Markdown("## <center>RT Canvas")
                gr.Markdown("<center>Basic Settings")
                f11_pro = gr.Textbox(placeholder="Prompt for image...", container=False, lines=2, max_lines=2)
                
                gr.Markdown("<center>Advanced Settings")
                f11_lra = gr.Dropdown(choices=sa.LoraList, value="None", container=False)
                f11_str = gr.Slider(value=0.89, minimum=0.0, maximum=1.0, step=0.001, label="Creativity Strength")
                f11_sed = gr.Number(value=0, minimum=0, label="Seed (0 for Random)")
                
                gr.Markdown("<center>Style Presets")
                f11_pre = gr.Dropdown(choices=styles_preset, value="None", container=False)
                f11_sty = gr.Dropdown(choices=["None"], value="None", container=False)
                f11_sub = gr.Button("Generate", variant="stop")
                f11_pre.change(fn=preset_select, inputs=[f11_pre], outputs=[f11_sty], show_progress="hidden")
            
            with gr.Column(variant="panel", scale=2) as input:
                import numpy as np
                f11_can = gr.Paint(
                    value=np.full((1024, 1024), 255, dtype=np.uint8),
                    type="pil",
                    container=False,
                    transforms=[],
                    sources=[],
                    canvas_size=(1024,1024),
                    height=610.94
                )
            
            with gr.Column(variant="panel", scale=2) as result:
                f11_res = gr.Gallery(container=False, height=610.94, elem_id="f11_res", show_share_button=False)
                f11_ram = gr.State([])
                f11_sub.click(
                    show_api=False,
                    scroll_to_output=True,
                    fn=f11_preprocess,
                    inputs=[f11_can, f11_pro, f11_lra, f11_str, f11_sed, f11_pre, f11_sty, f11_ram],
                    outputs=[f11_res]
                )

    with gr.Tab("Face Restore"):
        with gr.Row(equal_height=False):
            with gr.Column(variant="panel", scale=1) as menu:
                gr.Markdown("## <center>Face Restore")
                gr.Markdown("<center>Basic Settings")
                f12_img = gr.Image(type="filepath", height=279, sources=["upload"], label="Upload Image")
                f12_sub = gr.Button("Restore", variant="stop")
                
            with gr.Column(variant="panel", scale=3) as result:
                f12_res = gr.Gallery(container=False, height=939.938, elem_id="f7_res", show_share_button=False)
                f12_ram = gr.State([])
                f12_sub.click(
                    show_api=False,
                    scroll_to_output=True,
                    fn=f12_preprocess,
                    inputs=[f12_img, f12_ram],
                    outputs=[f12_res]
                )        
    
    with gr.Tab("Dual Consistency"):
        with gr.Row(equal_height=False):
            with gr.Column(variant="panel", scale=1) as menu:
                gr.Markdown("## <center>Dual Consistency")
                gr.Markdown("<center>Basic Settings")
                with gr.Row():
                    f13_fce = gr.Image(type="filepath", height=199, sources=["upload"], label="Face Image", min_width=48)
                    f13_stl = gr.Image(type="filepath", height=199, sources=["upload"], label="Style Image", min_width=48)
                f13_pro = gr.Textbox(placeholder="Prompt for image...", container=False)
                f13_neg = gr.Textbox(placeholder="Negative prompt...", container=False)
               
                gr.Markdown("<center>Advanced Settings")
                f13_siz = gr.Dropdown(choices=sa.SizeList, value=sa.SizeList[0], container=False)
                f13_fco = gr.Slider(value=1.2, minimum=0, maximum=2, step=0.05, label="Face Consistency")
                f13_sst = gr.Slider(value=0.7, minimum=0, maximum=1, step=0.05, label="Style Strength")
                f13_sed = gr.Number(value=0, minimum=0, label="Seed (0 for Random)")

                gr.Markdown("<center>Style Presets")
                f13_pre = gr.Dropdown(choices=styles_preset, value="None", container=False)
                f13_sty = gr.Dropdown(choices=["None"], value="None", container=False)
                f13_sub = gr.Button("Generate", variant="stop")
                f13_pre.change(fn=preset_select, inputs=[f13_pre], outputs=[f13_sty], show_progress="hidden")
                
            with gr.Column(variant="panel", scale=3) as result:
                f13_res = gr.Gallery(height=936.94, container=False, elem_id="f13_res", show_share_button=False)
                f13_ram = gr.State([])
                f13_sub.click(
                    show_api=False,
                    scroll_to_output=True,
                    fn=f13_preprocess,
                    inputs=[f13_fce, f13_stl, f13_pro, f13_neg, f13_siz, f13_fco, f13_sst, f13_sed, f13_pre, f13_sty, f13_ram],
                    outputs=[f13_res]
                )

    with gr.Tab("Face Identity"):
        with gr.Row(equal_height=False):
            with gr.Column(variant="panel", scale=1) as menu:
                gr.Markdown("## <center>Face Identity")
                gr.Markdown("<center>Basic Settings")
                f14_fce = gr.Image(type="filepath", height=199, sources=["upload"], label="Face Image")
                f14_pro = gr.Textbox(placeholder="Prompt for image...", container=False)
                f14_neg = gr.Textbox(placeholder="Negative prompt...", container=False)
               
                gr.Markdown("<center>Advanced Settings")
                f14_siz = gr.Dropdown(choices=sa.SizeList, value=sa.SizeList[0], container=False)
                f14_fco = gr.Slider(value=1.0, minimum=0, maximum=1, step=0.05, label="Face Consistency")
                f14_sed = gr.Number(value=0, minimum=0, label="Seed (0 for Random)")

                gr.Markdown("<center>Style Presets")
                f14_pre = gr.Dropdown(choices=styles_preset, value="None", container=False)
                f14_sty = gr.Dropdown(choices=["None"], value="None", container=False)
                f14_sub = gr.Button("Generate", variant="stop")
                f14_pre.change(fn=preset_select, inputs=[f14_pre], outputs=[f14_sty], show_progress="hidden")
                
            with gr.Column(variant="panel", scale=3) as result:
                f14_res = gr.Gallery(height=868.75, container=False, elem_id="f14_res", show_share_button=False)
                f14_ram = gr.State([])
                f14_sub.click(
                    show_api=False,
                    scroll_to_output=True,
                    fn=f14_preprocess,
                    inputs=[f14_fce, f14_pro, f14_neg, f14_siz, f14_fco, f14_sed, f14_pre, f14_sty, f14_ram],
                    outputs=[f14_res]
                )    

    with gr.Tab("Project Atelier"):
        with gr.Row(equal_height=False):
            with gr.Column(variant="panel", scale=1) as menu:
                gr.Markdown("## <center>Project Atelier")
                gr.Markdown("<center>Basic Settings")
                f15_pro = gr.Textbox(placeholder="Prompt for image...", container=False)
                f15_num = gr.Slider(value=4, minimum=1, maximum=8, step=1, label="Number of Images")
                
                gr.Markdown("<center>Image Guidance")
                f15_gdi = gr.Image(type="filepath", height=150, sources=["upload"], label="Guide Image")
                f15_gdt = gr.Dropdown(choices=sa.V4ControlList, value="None", container=False)
                f15_den = gr.Slider(value=0.95, minimum=0.1, maximum=1, step=0.01, label="Denoise Strength")

                gr.Markdown("<center>Advanced Settings")
                f15_v4 = gr.Dropdown(choices=sa.V4List, value="None", container=False)
                f15_mod = gr.Dropdown(choices=sa.AtelierList, value="Turbo", container=False)
                f15_siz = gr.Dropdown(choices=sa.SizeList, value=sa.SizeList[0], container=False)

                gr.Markdown("<center>Style Presets")
                f15_pre = gr.Dropdown(choices=styles_preset, value="None", container=False)
                f15_sty = gr.Dropdown(choices=["None"], value="None", container=False)
                f15_sub = gr.Button("Generate", variant="stop")
                f15_pre.change(fn=preset_select, inputs=[f15_pre], outputs=[f15_sty], show_progress="hidden")
                
            with gr.Column(variant="panel", scale=3) as result:
                f15_res = gr.Gallery(height=955.344, object_fit="contain", container=False, show_share_button=False, columns=4)
                f15_ram = gr.State([])
                f15_sub.click(
                    show_api=False,
                    scroll_to_output=True,
                    fn=f15_preprocess,
                    inputs=[f15_pro, f15_mod, f15_siz, f15_num, f15_gdi, f15_gdt, f15_den, f15_v4, f15_pre, f15_sty, f15_ram],
                    outputs=[f15_res]
                )    

    with gr.Tab("Project D3"):
        with gr.Row(equal_height=False):
            with gr.Column(variant="panel", scale=1) as menu:
                gr.Markdown("## <center>Project D3")
                gr.Markdown("<center>Basic Settings")
                f16_pro = gr.Textbox(placeholder="Prompt for image...", container=False, lines=5, max_lines=5)
                f16_sub = gr.Button("Generate", variant="stop")

            with gr.Column(variant="panel", scale=3) as result:
                f16_res = gr.Gallery(height=945.875, object_fit="contain", container=False, show_share_button=False)
                f16_ram = gr.State([])
                f16_sub.click(
                    show_api=False,
                    scroll_to_output=True,
                    fn=f16_preprocess,
                    inputs=[f16_pro, f16_ram],
                    outputs=[f16_res]
                )

if __name__ == "__main__":
    demo.launch(inbrowser=True)


def gradio_change_style(style):
    if style == "None":
        return gr.Dropdown(choices=["None"], value="None")
    else:
        styleList_name = f"{style}List"
        styleList = getattr(sa, styleList_name)
        return gr.Dropdown(choices=styleList, value=styleList[0])
    
def gradio_combine_strings(*args):
    combined = []
    for arg in args:
        if arg is not None and str(arg) != "None":
            combined.append(str(arg))
    return ', '.join(combined)

#     with gr.Tab("Image LUT Processor"):
#         lutFile = gr.Dropdown(choices=App.CubeList, value="Good Morning") label="LUT File")
#         with gr.Row():
#             lutImageInput3 = gr.Image(format="PNG") type=filepath") label="Input Image")
#             lutImageOutput2 = gr.Image(format="PNG") type=filepath") label="Output Image")
#         lutDescSubmit3 = gr.Button("Restore This Image")
#         lutDescSubmit3.click(
# show_api=False,
# scroll_to_output=True,fn=App.image_lut_processor, inputs=[lutImageInput3, lutFile], outputs=lutImageOutput2)

#     with gr.Tab("Prompt Designer"):
#         with gr.Row():
#             scene = gr.Dropdown(choices=App.Prompt["scene"], label="Scene Type") value=App.Prompt["scene"][0])
#             filter = gr.Dropdown(choices=App.Prompt[filter"], label=filter Type") value=App.Prompt[filter"][0])
#             camera = gr.Dropdown(choices=App.Prompt["camera"], label="Camera Type") value=App.Prompt["camera"][0])
#             material = gr.Dropdown(choices=App.Prompt["material"], label="Material Type") value=App.Prompt["material"][0])
#             perspective = gr.Dropdown(choices=App.Prompt["perspective"], label="Perspective Type") value=App.Prompt["perspective"][0])
        
#         with gr.Row():            
#             medium = gr.Dropdown(choices=App.Prompt["medium"], label="Art Medium") value=App.Prompt["medium"][0])
#             lighting = gr.Dropdown(choices=App.Prompt["lighting"], label="Lighting Option") value=App.Prompt["lighting"][0])
#             rendering = gr.Dropdown(choices=App.Prompt["rendering"], label="Rendering Engine") value=App.Prompt["rendering"][0])
#             artstyle = gr.Dropdown(choices=App.Prompt["artstyle"], label="Art Style") value=App.Prompt["artstyle"][0])
#             painter = gr.Dropdown(choices=App.Prompt["painter"], label="Popular Painter") value=App.Prompt["painter"][0])
        
#         pdPrompt = gr.Textbox(label="Original Prompt")
#         pdResult = gr.Textbox(label="Designed Prompt")
        
#         combine = gr.Button("Design Prompt")
#         combine.click(
# show_api=False,
# scroll_to_output=True,fn=combine_strings, 
#                       inputs=[pdPrompt, scene, filter, camera, material, perspective, medium, lighting, rendering, artstyle, painter],
#                       outputs=pdResult, show_progress='hidden')
    
#     #b.launch(inbrowser=True)