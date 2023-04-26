import pandas as pd
import cv2
import numpy as np

import time
from keras.models import model_from_json

import base64
import io
import dash
from dash import dcc, html
from dash_extensions.enrich import Output,  Input
from dash import dash_table as dct
import dash_bootstrap_components as dbc
from PIL import Image
from flask import Flask, Response
import os
import dash_daq as daq
import plotly.graph_objects as go

import plotly.io as plt_io

import plotly.express as px
import datetime
plt_io.templates["custom_dark"] = plt_io.templates["plotly_dark"]

plt_io.templates["custom_dark"]['layout']['paper_bgcolor'] = '#000000'
plt_io.templates["custom_dark"]['layout']['plot_bgcolor'] = '#000000'
plt_io.templates['custom_dark']['layout']['yaxis']['gridcolor'] = '#4f687d'
plt_io.templates['custom_dark']['layout']['xaxis']['gridcolor'] = '#4f687d'




theme = {
    'dark': True,
    'detail': '#007439',
    'primary': '#00EA64',
    'secondary': '#6E6E6E',
}

class VideoCamera(object):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    emotion_images_path = "assets/emotions"
    plutchic_d = {'271:285':'interest',
                    '286:300':'anticipation',
                    '301:315':'vigilance',
                    '316:330':'acceptance',
                    '331:345':'trust',
                    '346:360':'admiration',
                    '0:15':'serenity',
                    '16:30':'joy',
                    '31:45':'ecstasy',
                    '46:60':'distraction',
                    '61:75':'surprise',
                    '76:90':'amazement',
                    '256:270':'pensiveness',
                    '241:255':'sadness',
                    '226:240':'grief',
                    '211:225':'boredom',
                    '196:210':'disgust',
                    '181:195':'loathing',
                    '166:180':'annoyance',
                    '151:165':'anger',
                    '136:150':'rage',
                    '121:135':'apperhension',
                    '106:120':'fear',
                    '91:105':'terror',
                        }
    new_plot = {'0:45':'happiness',
                    '46:90':'surprise',
                    '91:130':'fear',
                    '131:155':'anger',
                    '156:180':'disgust',
                    '181:230':'sadness',
                    '270:310':'neutral',
                    
                        }
    angle_d = {-4:120,-3:140,-2:160,-1:215,0:270,1:310,2:30,3:80}
    
    angle_detect = {'0:45':2,
                    '46:90':3,
                    '91:130':-4,
                    '131:155':-3,
                    '156:180':-2,
                    '181:230':-1,
                    '270:300':0,
                    '301:330':1,
                    
                        }
    e_rate_color = [0,((0,128,0),(0,200,0)),
    ((0,0,128),(0,0,200)),((90,135,165),(138,210,255)),
    ((150,150,0),(255,255,0)),((155,100,0),(255,165,0)),((128,0,0),(200,0,0))]
    def __init__(self):
        json_file = open('fer_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.emotion = ''
        self.model.load_weights("fer_model.h5")
        print("Loaded model from disk")
        self.image_test = 'assets/ready.jpg'
        self.video = cv2.VideoCapture(0)
        self.nc_start =0
        self.server = Flask(__name__)
        self.server.add_url_rule('/video_feed', 'video_feed', view_func=self.video_feed)
        self.app = dash.Dash(__name__, server=self.server, external_stylesheets=[dbc.themes.CYBORG])
        self.app.layout = html.Div(
            [
                dbc.Tabs(
                    [
                        dbc.Tab(self.page_1(), tab_id="page_1",label="Start Form"),
                        dbc.Tab(self.page_2(), tab_id="page_2",label="Experiment"),
                        dbc.Tab(self.page_3(), tab_id="page_3",label="Results"),
                        ],
                    id="tabs",
                    active_tab="page_1",
                    ),
                dbc.Row(html.Div(id="content")),
                dbc.Row(html.Br(),
                        # dcc.Store(id='new_usr_str'),
                        ),
               
                ]
            )
        self.app.callback(Output("startBTN", "on"),
                         Output("test-img", "src"),
                        
                        Input("startBTN", "on"),
                          
                                                    
                          prevent_initial_call=True)(self.change_image_test)

        self.app.callback(Output("start-p", "childrens"),
                           
                          Input("startBTN", "on"),
                                                    
                          prevent_initial_call=True)(self.start)
        
        
        self.app.callback(Output("down", "n_clicks"),
                           
                          Input("down", "n_clicks"),
                                                    
                          prevent_initial_call=True)(self.down)
        
        self.app.callback(
                            Output("resultIMG1", "figure"),
                            Output("resultIMG2", "figure"),
                            Output("resultIMG3", "figure"),
                            Output("resultIMG4", "figure"),
                            Output("resultIMG5", "figure"),
                            Output("resultIMG6", "figure"),
                            Output("res-graph", "figure"),

                            Input("resultBTN", "on"),
                          
                                                    
                          prevent_initial_call=True)(self.result)
        self.app.callback(
                            Output("user-table", "data"),
                            Output("user-table", "columns"),
                            Output("sabt", "n_clicks"),
                            Input('name', 'value'),
                            Input('family', 'value'),
                            Input('e-dropdown', 'value'),
                            Input("sabt", "n_clicks"),
                          
                                                    
                          prevent_initial_call=True)(self.sabt)
    def sabt(self,name , family,emotion, n_clicks):

        if n_clicks>0:
            new_dict = {"0":emotion,"1":family,
                            "2":name}
            columns=[{"name": c, "id": str(i)} for i,c in enumerate(['Experiment','Family','Name'])]
            self.name = name
            self.family = family
            self.emotion = emotion
            return [new_dict],columns,0


    def down(self,n_click):
        if n_click>0:
            data = []
            d02 = {-4:0,-3:0,-2:0,-1:0,0:0,1:0,2:0,3:0}
            d25 = {-4:0,-3:0,-2:0,-1:0,0:0,1:0,2:0,3:0}
            d58 = {-4:0,-3:0,-2:0,-1:0,0:0,1:0,2:0,3:0}
            d811 = {-4:0,-3:0,-2:0,-1:0,0:0,1:0,2:0,3:0}
            d1114 = {-4:0,-3:0,-2:0,-1:0,0:0,1:0,2:0,3:0}
            d1416 = {-4:0,-3:0,-2:0,-1:0,0:0,1:0,2:0,3:0}
            for i in self.detect02:
                for k in self.angle_detect:
                    if int(i) in range(int(k.split(':')[0]),int(k.split(':')[1])):
                        d02[self.angle_detect[k]]+=1
                        break 
            
            for i in self.detect25:
                for k in self.angle_detect:
                    if int(i) in range(int(k.split(':')[0]),int(k.split(':')[1])):
                        d25[self.angle_detect[k]]+=1
                        break 
            
            for i in self.detect58:
                for k in self.angle_detect:
                    if int(i) in range(int(k.split(':')[0]),int(k.split(':')[1])):
                        d58[self.angle_detect[k]]+=1
                        break 

            
            for i in self.detect811:
                for k in self.angle_detect:
                    if int(i) in range(int(k.split(':')[0]),int(k.split(':')[1])):
                        d811[self.angle_detect[k]]+=1
                        break 

            for i in self.detect1114:
                for k in self.angle_detect:
                    if int(i) in range(int(k.split(':')[0]),int(k.split(':')[1])):
                        d1114[self.angle_detect[k]]+=1
                        break 

            for i in self.detect1416:
                for k in self.angle_detect:
                    if int(i) in range(int(k.split(':')[0]),int(k.split(':')[1])):
                        d1416[self.angle_detect[k]]+=1
                        break 
            d_list = [d02,d25,d58,d811,d1114,d1416]
            tags = ['t < 2s',' 2s < t < 5s',' 5s < t < 8s',' 8s < t < 11s',' 11s < t < 14s',' 14s < t < 16s']

            for i,d in enumerate(d_list):
                name_list = [self.name,self.family,self.emotion]
                res_list = list(d.values())
                t_list = name_list+res_list+[tags[i]]
                data.append(t_list)
            print(data)

            df = pd.DataFrame(data,columns=['name','family','test','fear','anger','disgust',
            'sadness','neutral','calm','happiness','surprise','time'])
            filename = 'reports/'+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'.xlsx'
            df.to_excel(filename,index=False)

        return 0
    def result(self,on):
        if on:
            d02 = {-4:0,-3:0,-2:0,-1:0,0:0,1:0,2:0,3:0}
            d25 = {-4:0,-3:0,-2:0,-1:0,0:0,1:0,2:0,3:0}
            d58 = {-4:0,-3:0,-2:0,-1:0,0:0,1:0,2:0,3:0}
            d811 = {-4:0,-3:0,-2:0,-1:0,0:0,1:0,2:0,3:0}
            d1114 = {-4:0,-3:0,-2:0,-1:0,0:0,1:0,2:0,3:0}
            d1416 = {-4:0,-3:0,-2:0,-1:0,0:0,1:0,2:0,3:0}
            for i in self.detect02:
                for k in self.angle_detect:
                    if int(i) in range(int(k.split(':')[0]),int(k.split(':')[1])):
                        d02[self.angle_detect[k]]+=1
                        break 
            
            for i in self.detect25:
                for k in self.angle_detect:
                    if int(i) in range(int(k.split(':')[0]),int(k.split(':')[1])):
                        d25[self.angle_detect[k]]+=1
                        break 
            
            for i in self.detect58:
                for k in self.angle_detect:
                    if int(i) in range(int(k.split(':')[0]),int(k.split(':')[1])):
                        d58[self.angle_detect[k]]+=1
                        break 

            
            for i in self.detect811:
                for k in self.angle_detect:
                    if int(i) in range(int(k.split(':')[0]),int(k.split(':')[1])):
                        d811[self.angle_detect[k]]+=1
                        break 

            for i in self.detect1114:
                for k in self.angle_detect:
                    if int(i) in range(int(k.split(':')[0]),int(k.split(':')[1])):
                        d1114[self.angle_detect[k]]+=1
                        break 

            for i in self.detect1416:
                for k in self.angle_detect:
                    if int(i) in range(int(k.split(':')[0]),int(k.split(':')[1])):
                        d1416[self.angle_detect[k]]+=1
                        break 

            




            russell1 = cv2.imread('assets/russell.png')
            russell2 = cv2.imread('assets/russell.png')
            russell3 = cv2.imread('assets/russell.png')
            russell4 = cv2.imread('assets/russell.png')
            russell5 = cv2.imread('assets/russell.png')
            russell6 = cv2.imread('assets/russell.png')
            
            length = 100
            p1=(250,250)
            ds = [0,d02,d25,d58,d811,d1114,d1416]
            imgs = [0,russell1,russell2,russell3,russell4,russell5,russell6]
            for c in range(1,7):
                for k in range(-4,4):
                    rate = ds[c][k]
                    angle= self.angle_d[k]
                    angle=-angle
                    p2 =  (int(p1[0] + length * np.cos(angle * 3.14 / 180.0)),
                    int(p1[1] + length * np.sin(angle * 3.14 / 180.0)))
                    divider = 2
                    if rate!=0:
                        for i in range(1,7):
                            if (rate//divider)<i:
                                
                                cv2.circle(imgs[c],p2,7*i,(self.e_rate_color[i][1][0],
                                self.e_rate_color[i][1][1],
                                self.e_rate_color[i][1][2],0.1*i),-1)
                                cv2.circle(imgs[c],p2,5*i,(self.e_rate_color[i][0][0],
                                self.e_rate_color[i][0][1],
                                self.e_rate_color[i][0][2],0.2*i),-1)
                                break
                        if (rate//divider)>=6:
                            i = 6
                            cv2.circle(imgs[c],p2,7*i,(self.e_rate_color[i][1][0],
                            self.e_rate_color[i][1][1],
                            self.e_rate_color[i][1][2],0.1*i),-1)
                            cv2.circle(imgs[c],p2,5*i,(self.e_rate_color[i][0][0],
                            self.e_rate_color[i][0][1],
                            self.e_rate_color[i][0][2],0.2*i),-1)


            



            img1 = px.imshow(russell1,width=450,height=450)
            
            img2 = px.imshow(russell2,width=450,height=450)
            img3 = px.imshow(russell3,width=450,height=450)
            img4 = px.imshow(russell4,width=450,height=450)
            img5 = px.imshow(russell5,width=450,height=450)
            img6 = px.imshow(russell6,width=450,height=450)
            img1.layout.template = 'custom_dark'
            img2.layout.template = 'custom_dark'
            img3.layout.template = 'custom_dark'
            img4.layout.template = 'custom_dark'
            img5.layout.template = 'custom_dark'
            img6.layout.template = 'custom_dark'
            
            img1.update_xaxes(showticklabels=False)
            img1.update_yaxes(showticklabels=False)

            img2.update_xaxes(showticklabels=False)
            img2.update_yaxes(showticklabels=False)

            img3.update_xaxes(showticklabels=False)
            img3.update_yaxes(showticklabels=False)

            img4.update_xaxes(showticklabels=False)
            img4.update_yaxes(showticklabels=False)

            img5.update_xaxes(showticklabels=False)
            img5.update_yaxes(showticklabels=False)

            img6.update_xaxes(showticklabels=False)
            img6.update_yaxes(showticklabels=False)
            dt = {-4:0,-3:0,-2:0,-1:0,0:0,1:0,2:0,3:0}
            for i in self.detections:
                for k in self.angle_detect:
                    if int(i) in range(int(k.split(':')[0]),int(k.split(':')[1])):
                        dt[self.angle_detect[k]]+=1
                        break 

            fig = go.Figure(data=[go.Bar(
                x=list(dt.keys()),
                y=list(dt.values()),
                marker_color=['red','orange','yellow','LightYellow','SkyBlue','LightSkyBlue','blue','green'] # marker color can be a single color value or an iterable
            )])
            fig.layout.template = 'custom_dark'

            fig.update_layout(title={
                'text': f"Rate : {list(dt.keys())[np.argmax(list(dt.values()))]}",
         'y':0.9, # new
         'x':0.5,
         'xanchor': 'center',
         'yanchor': 'top' # new

            })

            return img1,img2,img3,img4,img5,img6,fig
        else:
            fig = go.Figure(data=[go.Bar(
            )])
            fig.layout.template = 'custom_dark'

            fig.update_layout(title={
                'text': "Result?",
         'y':0.9, # new
         'x':0.5,
         'xanchor': 'center',
         'yanchor': 'top' # new

            })

            return fig
    def page_2(self):
        select_tab = dbc.Card(
                    dbc.CardBody(
                        [

                            dbc.Row(
                                html.Div([
            daq.PowerButton(
        on=False,
        theme='dark',
        id='startBTN',style={"background": "#000000",},
        className='dark-theme-control',
        size=100
    ),
            ],
            className="h-100 d-flex align-items-center justify-content-center")),
            dbc.Row(html.P(' ')),
            dbc.Row(
                                html.Div([
            

            html.Img(src=self.image_test,id='test-img',style=dict(
                    width='70%',
                   )),

            html.Img(src="/video_feed",style=dict(
                    width='20%',
                ))],
            className="h-100 d-flex align-items-center justify-content-center"))
                    
                                
                                ]),style={'background-color': 'black'})
        return select_tab
    
    
    

    def page_1(self):
        images = os.listdir(self.emotion_images_path)

        self.e_names = {}
        for image in images:
            name = image.split('.')[0]
            address = os.path.join(self.emotion_images_path,image)
            
            self.e_names[name]=address
        print(self.e_names.keys())
        select_tab = dbc.Card(
                    dbc.CardBody(
                        [

                            dbc.Row(
                                html.Div([
           dbc.Input(type="text", id='name',placeholder='Name',value='',style=dict(
                    width='41%',
                   direction='LTR'
                )),
            ],
            className="h-100 d-flex align-items-center justify-content-center")),
            dbc.Row(
                    html.P(' ',id='start-p')      
                ),
            dbc.Row(
                                html.Div([
           dbc.Input(type="text", id='family',value='' ,placeholder='Family',style=dict(
                    width='41%',
                    direction='LTR'
                   
                )),
            ],
            className="h-100 d-flex align-items-center justify-content-center")),
            dbc.Row(
                    html.P(' ')      
                ),
            dbc.Row(
                                html.Div([
           dcc.Dropdown(list(self.e_names.keys()), id='e-dropdown',style=dict(
                    width='73%',
                    verticalAlign="middle",
                    paddingLeft= '16%',
                    paddingRight= '0.75%',
                )),
            ],
            className="h-100 d-flex align-items-center justify-content-center"),align="center"
            
            
            ),

             dbc.Row(
                    html.P(' ')      
                ),
            dbc.Row(
                                html.Div([
            dbc.Button("Submit",id='sabt', n_clicks=0,style=dict(
                    width='20%',

                )),
            
            ],
            className="h-100 d-flex align-items-center justify-content-center")),


            dbc.Row(
                    html.P(' ')      
                ),
            dbc.Row(
                dct.DataTable(
                                id='user-table',
                                columns=[],
                                data=[],
                                style_cell={'overflowY': 'hidden','textOverflow': 'ellipsis',
                                            'maxWidth': 0, 'textAlign': 'center'
                                            },
                                editable=False)
            
            ,style={'width':'80%','paddingLeft': '20%',})



                                ]),style={'background-color': 'black'})
        return select_tab
    

    def page_3(self):
        select_tab = dbc.Card(
                    dbc.CardBody(
                        [

                            dbc.Row(
                                html.Div([
             daq.PowerButton(
        on=False,
        theme='dark',
        color=theme['primary'],
        id='resultBTN',style={"main-background": "#000000",},
        className='dark-theme-control',
        size=100
    ),
             
            ],
            className="h-100 d-flex align-items-center justify-content-center")),
            dbc.Row(html.P(' ')),
             dbc.Row(
                                html.Div([
                html.P(" ",style=dict(
                    width='25%', 
                    
                )),
            html.P("t < 2s",style=dict(
                    width='40%',
                    height='5%', 
                    
                )),
            html.P("2s < t <5s",style=dict(
                    width='40%',
                    height='5%', 
                    
                )),
            html.P("5s < t <8s",style=dict(
                    width='30%', 
                    height='5%',
                    
                )),
             
             
            ],
            className="h-100 d-flex align-items-center justify-content-center",style=dict(
                   height='5%', 
                    
                ))),
            
            dbc.Row(html.Div([
            dcc.Graph(id='resultIMG1',
            figure={
                'data': [
                    ],
                'layout': {
                    'paper_bgcolor':'#000000',
                    'plot_bgcolor':'#000000',
                }
                    },style=dict(
                                    width='30%',        
                   )
            ),
            dcc.Graph(id='resultIMG2',
            figure={
                'data': [
                    ],
                'layout': {
                    'paper_bgcolor':'#000000',
                    'plot_bgcolor':'#000000',
                }
                    },style=dict(
                                    width='30%',        
                   )
            ),
            dcc.Graph(id='resultIMG3',
            figure={
                'data': [
                    ],
                'layout': {
                    'paper_bgcolor':'#000000',
                    'plot_bgcolor':'#000000',
                }
                    },style=dict(
                                    width='30%',        
                   )
            )
],
            className="h-100 d-flex align-items-center justify-content-center")),


            dbc.Row(html.P(' ')),
             dbc.Row(
                                html.Div([
                html.P(" ",style=dict(
                    width='25%', 
                    
                )),
            html.P("8s < t < 11s",style=dict(
                    width='40%',
                    height='5%', 
                    
                )),
            html.P("11s < t <14s",style=dict(
                    width='40%',
                    height='5%', 
                    
                )),
            html.P("14s < t <16s",style=dict(
                    width='30%', 
                    height='5%',
                    
                )),
             
             
            ],
            className="h-100 d-flex align-items-center justify-content-center",style=dict(
                   height='5%', 
                    
                ))),
            
            dbc.Row(html.Div([
            
            
            dcc.Graph(id='resultIMG4',
            figure={
                'data': [
                    ],
                'layout': {
                    'paper_bgcolor':'#000000',
                    'plot_bgcolor':'#000000',
                }
                    },style=dict(
                                    width='30%',        
                   )
            ),
            dcc.Graph(id='resultIMG5',
            figure={
                'data': [
                    ],
                'layout': {
                    'paper_bgcolor':'#000000',
                    'plot_bgcolor':'#000000',
                }
                    },style=dict(
                                    width='30%',        
                   )
            ),
            dcc.Graph(id='resultIMG6',
            figure={
                'data': [
                    ],
                'layout': {
                    'paper_bgcolor':'#000000',
                    'plot_bgcolor':'#000000',
                }
                    },style=dict(
                                    width='30%',        
                   )
            )


],
            className="h-100 d-flex align-items-center justify-content-center")),
            
            dbc.Row(html.P(' ')),
            dbc.Row(html.P(' ')),
            dcc.Graph(id='res-graph',
   figure={
        'data': [
            ],
        'layout': {
            'paper_bgcolor':'#000000',
            'plot_bgcolor':'#000000',
        }
    }
),
            dbc.Row(
                                html.Div([
            dbc.Button("Save To Excel",id='down', n_clicks=0,className="btn btn-success",style=dict(
                    width='20%',

                )),
            
            ],
            className="h-100 d-flex align-items-center justify-content-center")),

                                
                                ]),style={'background-color': 'black'})
        return select_tab
    
    


    def __del__(self):
        self.video.release()
    
    def preprocess_input(self,x, v2=True):
        x = x.astype('float32')
        x = x / 255.0
        if v2:
            x = x - 0.5
            x = x * 2.0
        return x
        
    def get_frame(self):
        if True:
            success, image = self.video.read()
            if success:
                russell = cv2.imread('russell.png')
                
                russell = cv2.resize(russell,(600,500))
                
                image = cv2.resize(image,(600,500))
                            
                if self.nc_start:
                    if int(time.time()-self.start_time)>=16:
                        self.nc_start = False
                        self.detections= self.detect02+self.detect25+self.detect58+self.detect811+self.detect1114+self.detect1416
                        print(self.detections)


                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    if len(faces)>0:
                            (x, y, w, h) = faces[0]
                            n = 30
                            (x, y, w, h) = (x+n, y, w-(2*n), h)
                            face_gray = gray[y:y+h,x:x+w]
                            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            cv2.rectangle(image, (x, y+h - 35), (x+w, y+h), (0, 0, 255), cv2.FILLED)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            
                            length = 150
                            p1=(293,260)
                            cv2.circle(russell,p1,15,(0,0,0,.5),-1)
                            cv2.circle(russell,p1,10,(255,255,255),-1)

                            face_gray = cv2.resize(face_gray,(48,48))
                            
                            angle = self.model.predict(self.preprocess_input(face_gray).reshape((1,48,48,1)))
                            time_now = time.time()
                            dtime = int(time_now-self.start_time)
                            if dtime<2:
                                self.detect02.append(int(angle))
                            elif dtime<5:
                                self.detect25.append(int(angle))
                            elif dtime<8:
                                self.detect58.append(int(angle))
                            elif dtime<11:
                                self.detect811.append(int(angle))
                            elif dtime<14:
                                self.detect1114.append(int(angle))
                            elif dtime<16:
                                self.detect1416.append(int(angle))
                            else:
                                pass

                            
                            
                            new_plot = cv2.imread(r'E:\FER\assets\new_plot2\plot.png')
                                    
                            for k in self.new_plot.keys():
                                range_list = k.split(':')
                                if int(angle) in range(int(range_list[0]),int(range_list[1])):
                                    new_plot = cv2.imread(f'E:/FER/assets/new_plot2/{self.new_plot[k]}.png')

                                    cv2.putText(image, self.new_plot[k], (x + 6, y+h - 6), font, 1.0, (255, 255, 255), 1)
                                    break


                            angle=-angle
                            p2 =  (int(p1[0] + length * np.cos(angle * 3.14 / 180.0)),int(p1[1] + length * np.sin(angle * 3.14 / 180.0)))
                            cv2.line(russell,p1,p2,(0,0,0,0.5),10)
                            cv2.line(russell,p1,p2,(255,255,255),5)

                    else:
                        new_plot = cv2.imread(r'E:\FER\assets\new_plot2\plot.png')

                else:
                    new_plot = cv2.imread(r'E:\FER\assets\new_plot2\plot.png')


                vis = np.concatenate((image, russell,new_plot), axis=0)
                
                ret, jpeg = cv2.imencode('.jpg', vis)
                return jpeg.tobytes()
    



    

    def change_image_test(self,on):
        if on:
            print('----------------------------')
            print(self.emotion)
            self.image_test = self.e_names[self.emotion]

            
            with open(self.image_test, "rb") as image_file:
                buffer = io.BytesIO()

                img = Image.open(io.BytesIO(image_file.read()))# 
                new_img = img
                
                new_img.save(buffer, format="PNG")
                img_b64 = base64.b64encode(buffer.getvalue())
                img_b64 = img_b64.decode()
                img_data = "{}{}".format("data:image/jpg;base64, ", img_b64)
            return  1,img_data      


    def start(self,n_clicks):
        self.nc_start = n_clicks
        if n_clicks:
            self.start_time = time.time()
            self.detections = []
            self.detect02=[]
            self.detect25=[]
            self.detect58=[]
            self.detect811=[]
            self.detect1114=[]
            self.detect1416=[]
            

            self.video_feed()
        else:
            self.nc_start = False
    
    def stop(self,n_clicks):
        if not n_clicks:
            
            self.nc_start = 0

            

        return 0

    def work(self):
        while True:
            frame = self.get_frame()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    def video_feed(self):
        
        return Response(vc.work(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


vc = VideoCamera()




if __name__ == '__main__':
    vc.app.run_server(debug=False)