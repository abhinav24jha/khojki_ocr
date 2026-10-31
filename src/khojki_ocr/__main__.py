from pathlib import Path
import argparse
import hashlib
import json
import time


def main():
    p=argparse.ArgumentParser(description='Printed Khojki page OCR with UTF-8 output and an offline review page.')
    p.add_argument('source',type=Path,help='Page image or PDF')
    p.add_argument('--checkpoint',type=Path,default=Path(__file__).resolve().parents[2]/'models/printed-khojki.pt')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu',choices=['cpu','cuda','mps'])
    p.add_argument('--pages',type=int,nargs='+',help='One-based PDF page numbers; default all')
    p.add_argument('--pdf-native',action='store_true',help='Use the only embedded image per PDF page, preserving original scan resolution')
    p.add_argument('--region',type=int,nargs=4,metavar=('LEFT','TOP','RIGHT','BOTTOM'),help='Optional assisted body crop; default automatic')
    p.add_argument('--threads',type=int,default=2)
    args=p.parse_args()
    if not args.source.is_file():p.error('Source does not exist')
    if not args.checkpoint.is_file():p.error('Checkpoint does not exist; see README for the bundled model')
    if args.threads<1:p.error('--threads must be positive')
    if args.region and (min(args.region)<0 or args.region[2]<=args.region[0] or args.region[3]<=args.region[1]):p.error('Invalid crop coordinates')
    import torch
    from .recognize import Recognizer
    from .pipeline import page_images,process_page
    torch.set_num_threads(args.threads)
    start=time.perf_counter();recognizer=Recognizer(args.checkpoint,args.device);load=time.perf_counter()-start
    pages=[]
    for n,image in page_images(args.source,args.pages,args.pdf_native):
        if args.region and (args.region[2]>image.width or args.region[3]>image.height):p.error('Crop outside page')
        result=process_page(image,recognizer,args.output/f'page-{n:03d}',f'p{n:03d}',args.region)
        pages.append({'page':n,**{k:v for k,v in result.items() if k!='lines'}})
        print(f'Page {n}: {result["line_count"]} lines; {result["processing_seconds"]:.2f}s; {args.output/f"page-{n:03d}"/"review.html"}')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'run.json').write_text(json.dumps({'source':str(args.source),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),'device':args.device,'model_load_seconds':load,'pages':pages},indent=2))

if __name__=='__main__': main()
