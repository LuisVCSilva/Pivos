#!/usr/bin/python

from __future__ import print_function
from geopy.geocoders import Nominatim
from io import BytesIO
from joblib import Parallel, delayed
from ast import literal_eval
from scipy.spatial import distance

import random
import operator
import geopy
import geopy.distance as GlobalDistance
import math
import progressbar
import multiprocessing
import Image
import urllib
import os.path
import cv2
import numpy as np
import sys
import itertools
import csv
import base64

from math import log, exp, tan, atan, pi, ceil

EARTH_RADIUS = 6378137
EQUATOR_CIRCUMFERENCE = 2 * pi * EARTH_RADIUS
INITIAL_RESOLUTION = EQUATOR_CIRCUMFERENCE / 256.0
ORIGIN_SHIFT = EQUATOR_CIRCUMFERENCE / 2.0


SIZE = 640
ZOOM = 15

#parte do usuario
ID_GOOGLE = ""
CACHE_PATH = ""
RECORTE = [(0,0),(0,0)]
DELTA_FOTO = ""

RAW_PATH = ""
PRE_PATH = ""
SEG_PATH = ""
FINAL_PATH = ""

FLAG_VIDEO = False
FLAG_MODO = 0 #1 online ------ 2 offline
#fim da parte do usuario
FPS = 0

def CarregaPasta(folder):
 images = []
 for filename in os.listdir(folder):
  img = cv2.imread(os.path.join(folder,filename),0)
  if img is not None:
   images.append((img,filename[13:]))
 return images

#AIzaSyDcsE3oC8o6yb_shWerLcjVh15vwf-6OXY

def RecuperaRecorte (centro):
 url = "http://maps.googleapis.com/maps/api/staticmap?center="+str(centro[0])+","+str(centro[1])  +"&size="+str(SIZE)+"x"+str(SIZE)+"&zoom="+str(ZOOM)+"&maptype=satellite&key="+ID_GOOGLE
 #print(url)
 buffer = urllib.urlopen(url)
 if "text" in buffer.info().maintype:
  print("Erro encontrado na API Google Maps:\n[ERRO] " + buffer.read() + "\n")
  return -1
 else:
  raw = Image.open(BytesIO(buffer.read()))
  img = np.array(raw).copy()
  return img
 return -1

def Preprocessa(image):
 img = image

 cv2.imshow("0",img)
 cv2.waitKey(0)

 clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
 cl1 = clahe.apply(img)

 cv2.imshow("1",img)
 cv2.waitKey(0)

 
 v = np.median(img)
 img = cv2.fastNlMeansDenoising(src=img,dst=None,templateWindowSize=17,searchWindowSize=35,h=21)

 cv2.imshow("2",img)
 cv2.waitKey(0)


 kernel = np.array([
					[1,1,1,1,1],
					[0,0,1,1,1],
					[0,0,0,1,1],
					[0,0,0,1,1],
					[0,0,0,0,1]
					],dtype=np.uint8)

 img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

 np.rot90(kernel)
 img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

 np.rot90(kernel)
 img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


 cv2.imshow("3",img)
 cv2.waitKey(0)

 #cv2.imshow("Qtde de branco",img)
 #cv2.waitKey(0)

 kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

 img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel) 
 np.rot90(kernel)

 img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel) 
 np.rot90(kernel)

 img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel) 
 np.rot90(kernel)
 
 cv2.imshow("4",img)
 cv2.waitKey(0)


 #img = cv2.GaussianBlur(img,(5,5),0)
 #sigma = 0.5
 #lower = int(max(0, (1.0 - sigma) * v))
 #upper = int(min(255, (1.0 + sigma) * v))
 #edges = cv2.Canny(img,lower,upper)
 #img = edges

 #img = cv2.GaussianBlur(img,(5,5),0) 
 #img = cv2.GaussianBlur(img,(3,3),0)
 #th,img = cv2.threshold(img,32,128,cv2.THRESH_BINARY)


 #cv2.imshow("Qtde de branco",img)
 #cv2.waitKey(0)
 

 '''
 #(_,cnts,_) = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 #cnt = cnts[1]

 #img = cv2.GaussianBlur(img,(5,5),0)
 #epsilon = 0.1*cv2.arcLength(cnt,True)
 #approx = cv2.approxPolyDP(cnt,epsilon,True) 
 '''
 ret,mask = cv2.threshold(img,32,255,cv2.THRESH_BINARY)
 where = np.where(mask==255)
 mask = cv2.bitwise_and(mask,img)

 img = mask

 cv2.imshow("5",img)
 cv2.waitKey(0)


 return v,img

def DetectaCirculos (imagem,medianaImagem,minRaio,maxRaio,tolerancia):
 minDist = (minRaio+maxRaio)*0.5
 circles = cv2.HoughCircles(imagem,cv2.HOUGH_GRADIENT,1,minDist,param1=medianaImagem,param2=tolerancia,minRadius=minRaio,maxRadius=maxRaio)	
 return circles




def getLocalAtual (centroRecorte):
 centroRecorteStr = str(centroRecorte[0]) + ", " + str(centroRecorte[1])
 geolocator = Nominatim()
 try:
  localAtual = geolocator.reverse(centroRecorteStr).raw
 except Exception as e:
  print("[Erro]: Geocode falhou ao buscar recorte.")
  return -1
 return localAtual

def scanRecorte (localAtual,diretorioDestino):
 print ("Escaneando Recorte",localAtual)
 nomeArq = str(("Recorte " + str(str(localAtual[0]) + "," + str(localAtual[1])) + ".jpg"))
 if os.path.isfile(diretorioDestino+"/[RAW] " + nomeArq):
  print ("[CACHE] Carregando imagem do arquivo")
  rawImage = cv2.imread("[RAW] " + nomeArq)
  #imagem = Preprocessa(rawImage)
 else:
  print ("[WEB] Carregando imagem da web")  
  print ("Salvando em " + diretorioDestino+"/[RAW] " + nomeArq)

  imagem = RecuperaRecorte(localAtual)
  cv2.imwrite(diretorioDestino+"/[RAW] " + nomeArq,imagem)
  rawImage = cv2.imread(diretorioDestino+"/[RAW] " + nomeArq)
  #imagem = Preprocessa(rawImage)
 return (rawImage,nomeArq[10:])
 #return (rawImage,imagem,nomeArq[9:])

def Segmenta (rawImage,imagem,medianaImagem,nomeArq,diretorioImagensPreprocessadas,diretorioImagensSegmentadas,diretorioPivos):
 circles = DetectaCirculos(imagem,medianaImagem,60,300,80)
 cv2.imwrite(diretorioImagensPreprocessadas+"/[PRE_RAW] " + nomeArq,imagem)
 if len(imagem.shape)!=3:
  imagem = cv2.cvtColor(imagem,cv2.COLOR_GRAY2RGB)

 if len(rawImage.shape)!=3:
  rawImage = cv2.cvtColor(rawImage,cv2.COLOR_GRAY2RGB)

 if circles is not None:
  circles = np.round(circles[0, :]).astype("int")
  for (x, y, r) in circles:

   aux = imagem
   mask = np.zeros(aux.shape, np.uint8)
   cv2.circle(mask, (x, y), int(r*0.75),(255,255,255), -1)
   where = np.where(mask==(255,255,255))
   mask = cv2.bitwise_and(mask,aux)
   brancura = np.sum(mask/255.0)/(math.pi*r*r)#porcentagem de branco do circulo
   #print(brancura)
   #cv2.imshow(str(brancura),mask)
   #cv2.waitKey(0)
   #cv2.destroyAllWindows()

   if brancura>0.15:#detectando falso pivo pela qtde de branco no interior
    np.setdiff1d(circles,[(x,y,r)])
    continue

   cv2.rectangle(rawImage,(x-r, y-r), (x + r, y + r), (0, 255, 255), 4)
   cv2.rectangle(imagem,(x-r, y-r), (x + r, y + r), (0, 255, 255), 4)

   
   cv2.circle(imagem, (x, y), r, (0, 255, 0), 4)
   cv2.circle(rawImage, (x, y), r, (0, 255, 0), 4)

   cv2.rectangle(rawImage, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
   cv2.rectangle(imagem, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
      
   cv2.imwrite(diretorioPivos+"/[PIV] " + nomeArq,rawImage)
   corte = rawImage[(y-r-10):(y+r+10),(x-r-10):(x+r+10)]

 cv2.imwrite(diretorioImagensSegmentadas+"/[SEG] " + nomeArq,rawImage)
 return circles,brancura if circles is not None else []

def pixels2latlon (lat,lon,px,py,r):
 return lat+(-0.000012384091*px),lon+(0.000011732301*py),4.488189*r

def ModoOnline (p1,p4,decrescesor,diretorioDestino):
 print("Recuperando imagens da web dentro do retangulo: ",(p1,p4))
 i=p1[0]
 #for i in np.arange(p4[0],p1[0],decrescesor):
 while i>=p4[0]:
  j = p4[1]
  #for j in np.arange(p1[1],p4[1],decrescesor):
  while j>=p1[1]:
   print("-----------------------------------------------------------------------------")
   coordenadaRecorte = (round(i,6),round(j,6))
   #print("Recuperando recorte",coordenadaRecorte)
   #raw,pre,nome = scanRecorte(coordenadaRecorte)
   raw,nome = scanRecorte(coordenadaRecorte,diretorioDestino)
   j -= decrescesor
  i -= decrescesor
 return 0 




def ModoOffline (diretorioRaw,diretorioPre,diretorioSeg,diretorioFinal,createVideo,fps):
 pivos = []
 print("Realizando deteccao de pivos em modo offline...")
 caminhoAlvo = os.path.dirname(os.path.realpath(__file__))
 qtdeArqs = len(os.listdir(caminhoAlvo + "/" + diretorioRaw))
 print("Utilizando",qtdeArqs,"imagens do diretorio: " + caminhoAlvo + "/" + diretorioRaw)
 print("Realizando gravacoes de imagens pre-processadas no diretorio: " + caminhoAlvo + "/" + diretorioPre)
 print("Realizando gravacoes de imagens segmentadas no diretorio: " + caminhoAlvo + "/" + diretorioSeg)
 if createVideo:
  print("Agendando criacao de video com as imagens com",fps,"quadros por segundo...")
 imagensOriginais = CarregaPasta(caminhoAlvo + "/" + diretorioRaw)
 for elemento in zip(imagensOriginais,range(1,len(imagensOriginais)+1)):
  indice = elemento[1]
  imagem = elemento[0]
  nomeImagem = imagem[1]

  coordenadaRecorte = literal_eval(("("+nomeImagem[:-4]+")"))
  imagemOriginal = imagem[0]

  medianaImagem,imagemPreprocessada = Preprocessa(imagemOriginal)
  saida,brancura = Segmenta(imagemOriginal,imagemPreprocessada,medianaImagem,nomeImagem,diretorioPre,diretorioSeg,diretorioFinal)
  if saida is not None:
   print("".join(["-" for x in range(0,len("Realizando deteccao de pivos em modo offline..."))]))
   print("\nRecorte",indice)
   print("Coordenada no recorte:",coordenadaRecorte)
   local = -1#getLocalAtual(coordenadaRecorte)
   if local==-1:
    #print ("Erro na recuperacao do local")
    local = "Erro"
   else:
    print()
    #print("Pais: ",local["address"]["country"])   
    #print("Estado: ",local["address"]["state"])   
    #print("Mesoregiao: ",local["address"]["state_district"])
    #print("Microregiao: ",local["address"]["county"])
    #print("\n\nRelacao dos pivos no Recorte:")
   #print("{: <10} {: <39} {: <10}".format(*("#ID","Posicao","Raio")))
   for c in zip(saida,range(1,len(saida)+1)):
    coordCirc = ""
    latn, lonn, raio = pixels2latlon(coordenadaRecorte[0],coordenadaRecorte[1],c[0][0],c[0][1],c[0][2])
    #print("{: <10} {: <39} {: <10}".format(*(c[1],(latn,lonn),raio)))
    pivos.append(("id","coordenada do pivo","raio",brancura,diretorioSeg + "/[SEG] " + nomeImagem))
   #print("\nTotal =",len(saida))
                                        
   with open('saida.csv','wb') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['id','posicao','raio','brancura','image'])
    for row in pivos:
     csv_out.writerow(row)

 if createVideo:
  os.system("ffmpeg -framerate "+ str(fps) +" -pattern_type glob -i '"+diretorioRaw+"/*.jpg' "+diretorioRaw+"/video.mp4 -y")
  os.system("ffmpeg -framerate "+ str(fps) +" -pattern_type glob -i '"+diretorioPre+"/*.jpg' "+diretorioPre+"/video.mp4 -y")
  os.system("ffmpeg -framerate "+ str(fps) +" -pattern_type glob -i '"+diretorioSeg+"/*.jpg' "+diretorioSeg+"/video.mp4 -y")

 print("\n\nLista dos pivos: ")
 for pivo in pivos:
  print (str(pivo)+"\n")
 return 0





tokensAjuda = (["--help","-h"],"Exibe esta mensagem de ajuda e sai do programa.")
tokensSobre = (["--about","-a"],"Exibe info sobre a autoria e proposito deste programa e sai do programa")

tokensModoOnline = (["--online","-on"],"Recupera imagens de satelite usando a Google Maps API")
tokensModoOffline = (["--offline","-off"],"Realiza a segmentacao e contagens de pivos em imagens de um diretorio")

tokensIdGoogle = (["--idGoogle","-id"],"Define um ID da API Google necessaria para a recuperacao das imagens no Modo Online.")
tokensScanAreas = (["--area","-aa"],"Delimita o recorte do globo que sera buscada.")
tokensDeltaFoto = (["--deltaFoto","-df"],"Define um intervalo de distancia entre as imagens.")


tokensDiretorioCache = (["--cachePath","-chc"],"Especifica o diretorio no qual as imagens recuperadas via Web serao gravadas")
tokensDiretorioRaw = (["--rawPath","-raw"],"Especifica o diretorio no qual as imagens originais serao gravadas/lidas")
tokensDiretorioPre = (["--prePath","-pre"],"Especifica o diretorio no qual as imagens preprocessadas serao gravadas/lidas")
tokensDiretorioSeg = (["--segPath","-seg"],"Especifica o diretorio no qual as imagens segmentadas serao gravadas/lidas")
tokensDiretorioFinal = (["--finalPath","-fin"],"Especifica o diretorio no qual as imagens finais serao gravadas/lidas")


tokensGeraVideo = (["--createVideo","-cv"],"Sinaliza se um video sera gerado utilizando as imagens recuperadas/geradas")
tokensFps = (["--fps","-fps"],"Define a quantidade de quadros contidos em um segundo de video gerado")

tokens = [\
tokensAjuda,\
tokensSobre,\
tokensModoOnline,\
tokensModoOffline,\
tokensIdGoogle,\
tokensScanAreas,\
tokensDeltaFoto,\
tokensDiretorioCache,\
tokensDiretorioRaw,\
tokensDiretorioPre,\
tokensDiretorioSeg,\
tokensDiretorioFinal,\
tokensGeraVideo,\
tokensFps\
]	


def msgAjuda ():
 print("Uso: " + sys.argv[0] + " [argumentos]")
 comandos = [str(token[0]).replace("'","")[1:-1] for token in tokens]
 descricao = [token[1] for token in tokens]

 for comando,descricao in zip(comandos,descricao):
  print("{: <20} {: <20}".format(*(comando,descricao)))

 print("\n\nObservacoes: ")
 print("1 -> Por razoes de seguranca, salve a chave da API Google em um arquivo e passe-a ao script usando 'cat', e.g: \"--idGoogle $(cat key.txt)\"")
 print("2 -> Defina um \"delta-foto\" pequeno caso queira gerar videos")
 print("3 -> O argumento de area deve conter duas coordenadas na forma \"latitude,longitude\" separados por um espaco. Estas duas coordenadas serao as arestas geradoras de um retangulo projetado na superficie do globo.")
 return 0


def msgSobre ():
 print("\nPrograma desenvolvido na disciplina de Topicos 3 para busca e deteccao de pivos irrigacao em imagens de satelite.x"+\
"\n\nAutores:\n"+\
"Discente: Luis Vinicius Costa Silva\n"+\
"Docente: Dr. Marcos Aurelio Batista\n\n"+\
"UFG - Regional Catalao - 2017.2 - Ciencia da Computacao\n\n\n"+\
"Disponivel em: http://github.com/LuisVCSilva/t3 sob a licenca GPL."
)
 return 0


def AnalisaExpressao (expressao,checador):
 if len(expressao)>1:
  for arg in expressao[1:]:
   if arg in [y for x in [token[0] for token in tokens] for y in x]:
    if arg in expressao:
     expressao = checador(expressao[expressao.index(arg):])
   elif expressao:
    break
 else:
  print ("[Erro] Nenhum argumento presente na chamada do programa...")
  msgAjuda()
 return 0

def ChecadorGeral (expressao):

 def ChecadorModoOffline (expressao):
  global FLAG_MODO
  FLAG_MODO = 2
  for arg in expressao:
   if arg in tokensDiretorioRaw[0]:
    #print("Recuperando imagens originais de: " + expressao[expressao.index(arg)+1])
    global RAW_PATH
    RAW_PATH = expressao[expressao.index(arg)+1]
   if arg in tokensDiretorioPre[0]:
    #print("Gravando imagens preprocessadas de: " + expressao[expressao.index(arg)+1])
    global PRE_PATH
    PRE_PATH = expressao[expressao.index(arg)+1]
   if arg in tokensDiretorioSeg[0]:
    #print("Gravando imagens segmentadas de: " + expressao[expressao.index(arg)+1])
    global SEG_PATH
    SEG_PATH = expressao[expressao.index(arg)+1]
   if arg in tokensDiretorioFinal[0]:
    #print("Gravando imagens com pivos em: " + expressao[expressao.index(arg)+1])
    global FINAL_PATH
    FINAL_PATH = expressao[expressao.index(arg)+1]
   if arg in tokensGeraVideo[0]:
    global FLAG_VIDEO
    FLAG_VIDEO = True
   if arg in tokensFps[0]:
    global FPS
    FPS = expressao[expressao.index(arg)+1]
  return expressao[expressao.index(arg)+1:]

 def ChecadorModoOnline (expressao):
  global FLAG_MODO
  FLAG_MODO = 1
  for arg in expressao:
   if arg in tokensIdGoogle[0]:
    #print("IdAPI = " + expressao[expressao.index(arg)+1])
    global ID_GOOGLE
    ID_GOOGLE = expressao[expressao.index(arg)+1]
   if arg in tokensDiretorioCache[0]:
    #print("Cache = " + expressao[expressao.index(arg)+1])
    global CACHE_PATH
    CACHE_PATH = expressao[expressao.index(arg)+1]
   if arg in tokensScanAreas[0]:
    #print ("Recorte = " + expressao[expressao.index(arg)+1])
    global RECORTE
    RECORTE = ("[(" + expressao[expressao.index(arg)+1] + ")," + "(" + expressao[expressao.index(arg)+2] + ")]")
   if arg in tokensDeltaFoto[0]:
    global DELTA_FOTO
    DELTA_FOTO = expressao[expressao.index(arg)+1]
    print ("Delta Foto =",DELTA_FOTO)

  return expressao[expressao.index(arg)+1:]


 for arg in expressao:
  if arg in tokensAjuda[0]:
   msgAjuda()
  elif arg in tokensSobre[0]:
   msgSobre()
  elif arg in tokensModoOnline[0]:
   #print("Modo online")
   AnalisaExpressao(expressao[expressao.index(arg):],ChecadorModoOnline)
  elif arg in tokensModoOffline[0]:
   #print("Modo offline")
   AnalisaExpressao(expressao[expressao.index(arg):],ChecadorModoOffline)
 return expressao[expressao.index(arg):]

def main():
 AnalisaExpressao(sys.argv,ChecadorGeral)
 print(DELTA_FOTO)
 if FLAG_MODO==1:
  ModoOnline(literal_eval(RECORTE)[0],literal_eval(RECORTE)[1],float(DELTA_FOTO),str(CACHE_PATH))
 if FLAG_MODO==2:
  ModoOffline(str(RAW_PATH),str(PRE_PATH),str(SEG_PATH),str(FINAL_PATH),FLAG_VIDEO,int(FPS))

#-16.933107, -47.678304
#-17.035908, -47.572012 
if __name__ == '__main__':
 main()
