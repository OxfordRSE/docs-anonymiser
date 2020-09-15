#!/usr/bin/env python3

from __future__ import division, print_function
from wand.image import Image, Color
from PIL import Image as PI
import pytesseract
import argparse
import io
import cv2
import re
import os
import sys
import numpy as np
from skimage.transform import radon
from skimage.morphology import disk, closing
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pycorenlp import StanfordCoreNLP
try:
    # More accurate peak finding from https://gist.github.com/endolith/255291#file-parabolic-py
    from parabolic import parabolic

    def argmax(x):
        return parabolic(x, np.argmax(x))[0]
except ImportError:
    from numpy import argmax

nlp = StanfordCoreNLP('http://localhost:9000')
#from matplotlib.mlab import rms_flat
import spacy
nlp_sp = spacy.load("en")


def add_rectangle_to_image(image, rect):
    if len(image.shape)==2:
        color = 255
    else:
        color = (255, 255, 255)
    image = cv2.rectangle(image, tuple(rect[0:2]), (rect[0] + rect[2], rect[1] + rect[3]), color, thickness=-1)
    return image


def preprocess_for_image(gray, blur):
    # check to see if we should apply thresholding to preprocess
    if blur:
        gray = cv2.medianBlur(gray, 3)
        selem = disk(1)
        gray = closing(gray, selem)

    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10)

    return gray


def rotation_spacing(I):
    # -*- coding: utf-8 -*-
    """
    Automatically detect rotation and line spacing of an image of text using
    Radon transform
    If image is rotated by the inverse of the output, the lines will be
    horizontal (though they may be upside-down depending on the original image)
    It doesn't work with black borders
    """

    # Load file, converting to grayscale
    #I = asarray(Image.open(filename).convert('L'))
    I = I - np.mean(I)  # Demean; make the brightness extend above and below zero

    # Do the radon transform and display the result
    sinogram = radon(I)

    # Find the RMS value of each row and find "busiest" rotation,
    # where the transform is lined up perfectly with the alternating dark
    # text and white lines
    r = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()])
    rotation = argmax(r)
    if rotation < 45:
        rotation = rotation + 90
    print('Rotation: {:.2f} degrees'.format(90 - rotation))
    return rotation


def ner_extraction(text):
    output = nlp.annotate(text, properties={
        'annotators': 'ner',
        'outputFormat': 'json'
    })
    return output


def cleanup(token, lower = True):
    if lower:
       token = token.lower()
    return token.strip()


def blank_word(word, conv_img, excl=0):
    rect = [int(word[6]), int(word[7]), int(word[8])-excl, int(word[9])]

    conv_img = add_rectangle_to_image(conv_img, rect)
    return(conv_img)

def templates(term):
    match = False

    if re.match(r"[^@]+@[^@]+", term) or \
        re.match("^\s*[a-zA-Z]{2}(?:\s*\d\s*){6}[a-zA-Z]?\s*$", term) or \
        re.match("^0\d{4,10}$", term) or re.match("^\d{1}[A-Z]{2}$", term):

        match = True

    return(match)


def find_names(filename, rot_fl=0, blur=0):
    # load the example image and convert it to grayscale

    req_image = []
    conv_img_list = []
    gray_list = []
    search_terms = []
    doc_text = ''

    with Image(filename=filename, resolution=300) as image_jpeg:
        image_jpeg.compression_quality = 99
        image_jpeg = image_jpeg.convert('jpeg')

        for img in image_jpeg.sequence:
            with Image(image=img) as img_page:
                img_page.background_color = Color('white')
                img_page.alpha_channel = 'remove'
                req_image.append(img_page.make_blob('jpeg'))
    image_jpeg.destroy()

    for index, img in enumerate(req_image):
        # txt = pytesseract.image_to_string(PI.open(io.BytesIO(img)))
        conv_img = PI.open(io.BytesIO(img))
        conv_img = np.asarray(conv_img, dtype=np.uint8)

        if len(conv_img.shape) == 3:
            #conv_img = cv2.cvtColor(conv_img, cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(conv_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = conv_img
        gray = preprocess_for_image(gray, blur)

        # Rotate images
        if rot_fl == 1:
            rot = rotation_spacing(gray)
        else:
            rot = 90.0

        rows, cols = gray.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90 - rot, 1)
        gray = cv2.warpAffine(gray, M, (cols, rows))
        conv_img = cv2.warpAffine(conv_img, M, (cols, rows))
        # page_text = pytesseract.image_to_string(gray, config='--psm 11 -c textord_heavy_nr=1')
        # print(page_text)

        #doc_text = doc_text + page_text

        conv_img_list.append(conv_img)
        gray_list.append(gray)
        # cv2.imwrite(filename + str(index) + '.jpg', gray)

        # NLP analysis
        #nlp_result = ner_extraction(page_text)

        #nlp_result_sp = nlp_sp(page_text)
        #labels = set([w.label_ for w in nlp_result_sp.ents])
        #in_labels = ['PERSON', 'ORG', 'GPE', 'LOC', 'FAC']

        #others =[]

        #found_name = False
        #for sen in nlp_result["sentences"]:
        #    for tok in sen['tokens']:
        #        print('Stanford tok', tok)
        #        if tok['ner'] == 'PERSON' and not found_name and tok["word"] not in search_terms:
        #            print('Name:', tok["word"])
        #            search_terms.append(tok["word"])

        #print('after 1st', search_terms)

        #for tok in nlp_result_sp:
        #    print('spacy tok', tok.text, tok.lemma_, tok.pos_, tok.tag_)
        #    if tok.tag == 'PRP' and tok.text not in search_terms:
        #        print('PRP:', tok.text)

        #for ent in nlp_result_sp.ents:
        #    print('spacy ent', ent.text, ent.label_)


    print('SEARCH TERMS:', search_terms)

    return search_terms, conv_img_list, gray_list


def draw_boxes(search_terms, conv_img_list, gray_list, dir, filename):
    out_file = os.path.join(dir, filename[:-4] + "_anon.pdf")
    pp = PdfPages(out_file)

    for ind in range(len(conv_img_list)):
        print('processing page', ind)
        conv_img = conv_img_list[ind]
        gray = gray_list[ind]

        boxes = pytesseract.image_to_data(gray, config='--psm 12 -c textord_heavy_nr=1')
        # -c "textord_heavy_nr"=1 -c "textord_space_size_is_variable"=1
        #print(boxes)
        lines = boxes.split('\n')
        words = [x.split('\t') for x in lines]


        for i in range(len(lines)):
            if len(words[i]) == 12:
                if (words[i][11].lower() in search_terms):
                    conv_img = blank_word(words[i], conv_img)

                if words[i][11].lower() in ['man', 'woman']:
                    conv_img = blank_word(words[i], conv_img)

                if words[i][11].lower() in ['ms', 'miss', 'mrs', 'mr']:
                    conv_img = blank_word(words[i], conv_img)

                if words[i][11].lower() in ['she', 'her', 'hers', 'he', 'him', 'his']:
                    conv_img = blank_word(words[i], conv_img)

        imgplot = plt.figure(figsize=(8.27, 11.69), dpi=300)
        ax = imgplot.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False, aspect=1)
        if len(conv_img.shape) == 2:
            plt.imshow(conv_img, cmap='gray')
        else:
            plt.imshow(conv_img)
        ax.set_xticks([])
        ax.set_yticks([])
        pp.savefig()
        plt.close(imgplot)

        # cv2.imwrite("/Users/violetakovacheva/Documents/Annonimise_Samples/test.png", np.array(conv_img, dtype=np.uint8))
        #cv2.imshow("Output", gray)
        #cv2.waitKey(0)

    pp.close()


parser = argparse.ArgumentParser(description='Anonymise PDF documents.')

parser.add_argument('dir', nargs='?', help="Directory with PDFs to process")
parser.add_argument('skew', nargs='?', default=0, help="Skewness flag. Put 1 to correct skewness, 0 otherwise")
parser.add_argument('blur', nargs='?', default=0, help="Blurring flag. Put 1 to correct noise, 0 otherwise")
#filename = "/Users/violetakovacheva/Documents/Anonymise_Samples/Example One Page CV.pdf"


args = parser.parse_args()
if not args.dir:
        print('--Folder name is mandatory')
        sys.exit(1)

directory = os.fsencode(args.dir)

outdir = os.path.join(args.dir, 'Anonymised')
if not os.path.exists(outdir):
    os.makedirs(outdir)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".pdf"):
        print('processing file', filename)
        [search_terms, conv_img_list, gray_list] = find_names(os.path.join(args.dir, filename), args.skew, args.blur)
        names = filename.split(" - Application")[0].split(" ")
        names = [n.replace(',', '').lower() for n in names]
        names_possessive = [n + "'s" for n in names]
        print('NAMES:', names)
        search_terms = search_terms + names + names_possessive
        draw_boxes(search_terms, conv_img_list, gray_list, outdir, filename)
