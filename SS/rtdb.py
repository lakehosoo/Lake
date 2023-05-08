import pandas as pd 
import requests 
import json 

def get_rtdb_data(TagNames:list, StartTime:str, EndTime:str , SampleType:str , Frequency:int=600) -> pd.DataFrame: 
    # Server Setting 
    url = "http://s2appsvr/rtdbservice/PHDService.svc/getdata" 
    headers = {"Content-type" : "application/json", 
               "Accept" : "application/json"} 

    result_df = pd.DataFrame() 
    for i, TagName in enumerate(TagNames):         
        data = { 
                "TagName": TagName, 
                "StartTime": StartTime, 
                "EndTime": EndTime, 
                "SampleType" : SampleType, 
                "MaxRows": 100000000,    # 조회할 데이터 수량 (Default : 400) 
                "Frequency": Frequency             
                } 
        response = requests.post(url, headers = headers, data = json.dumps(data)) 
        df = pd.DataFrame.from_dict(json.loads(response.text)) 

        count, skip_flag = 0, 0 
        while pd.isnull(df.loc[0, 'TagName']): 
            response = requests.post(url, headers = headers, data = json.dumps(data))                                                 
            df = pd.DataFrame.from_dict(response.json()) 
            count += 1 
            if count == 10:                 
                skip_flag = 1 
                break 
        if skip_flag: 
            print(f'{TagName}은 불러올 수 없습니다. ({count}회 시도하였으나 실패)') 
            continue 

        df = df.loc[:, ['TimeStamp', 'TagValue']].rename(columns={'TagValue':TagName, 'TimeStamp':'DateTime'}) 
        df = df.astype({TagName:'float', 'DateTime':'datetime64'}) 
        
        if result_df.empty: 
            result_df = df 
        else: 
            result_df = result_df.merge(df, on='DateTime', how='outer')
        
    return result_df 

def get_rtdb_definition(TagNames:list) -> pd.DataFrame: 
    url = "http://s2appsvr/rtdbservice/PHDService.svc/GetTagDfn" 
    headers = {"Content-type" : "application/json", 
               "Accept" : "application/json"} 

    for i, TagName in enumerate(TagNames): 
        data = { "TagName": TagName, }
        
        response = requests.post(url, headers = headers, data = json.dumps(data)) 
        tag_definition = pd.DataFrame(json.loads(response.text), index=[0]) 

        if i==0: 
            result_df = tag_definition 
        else: 
            result_df = pd.concat([result_df, tag_definition], ignore_index=True) 

        return result_df 

def put_rtdb_data(TagName:str, TimeStamp:str, Value:float) -> None: 
    url = "http://s2appsvr/rtdbservice/PHDService.svc/PutData" 
    headers = {"Content-type" : "application/json", 
               "Accept" : "application/json"} 
    data = { 
        'TagName' : TagName, 
        'TimeStamp' : TimeStamp, 
        'Value' : Value          
    } 

    response = requests.post(url, headers = headers, data = json.dumps(data)) 
