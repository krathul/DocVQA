import hydra
from omegaconf import DictConfig, OmegaConf
import gradio as gr
from eom import Eom


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    Interface=gr.Blocks()
    with Interface:
        with gr.Blocks(title='Demo') as demo:
            gr.Markdown("<h1><center>Document Analyzer </center></h1>")

            with gr.Row():
                with gr.Column():
                    input=[gr.File(label='Select the pdf'),gr.Textbox(label='Enter search query')]
                    btn = gr.Button("GO")
                with gr.Column():        
                    output=[gr.Textbox(label='page number and your answer'),\
                            gr.Gallery(label="Generated images", show_label=False, elem_id="gallery").style(grid=[2], height="auto")]
            eom_ana =  Eom(cfg)     
            btn.click(fn=eom_ana,inputs=input,outputs=output)
        
    Interface.launch(share=True)
    

if __name__ == "__main__":
    my_app()