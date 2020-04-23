import requests
import calendar

def download_airports():
    cookies = {
        'ASPSESSIONIDCAQCSDSS': 'GOBIGIDBLGILGICKLFHKIHMN',
        '__utmt_ritaTracker': '1',
        '__utmt_GSA_CP': '1',
        '__utma': '261918792.554646962.1504352085.1504352085.1504352085.1',
        '__utmb': '261918792.2.10.1504352085',
        '__utmc': '261918792',
        '__utmz': '261918792.1504352085.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none)',
    }
    headers = {
        'Origin': 'https://www.transtats.bts.gov',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.8',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.101 Safari/537.36',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Cache-Control': 'max-age=0',
        'Referer': 'https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=288&DB_Short_Name=Aviation%20Support%20Tables',
        'Connection': 'keep-alive',

    }



    params = (

        ('Table_ID', '288'),

        ('Has_Group', '0'),

        ('Is_Zipped', '0'),

    )



    data = [

        ('UserTableName', 'Master_Coordinate'),

        ('DBShortName', 'Aviation_Support_Tables'),

        ('RawDataTable', 'T_MASTER_CORD'),

        ('sqlstr', ' SELECT AIRPORT_ID,AIRPORT,DISPLAY_AIRPORT_NAME,DISPLAY_AIRPORT_CITY_NAME_FULL,LATITUDE,LONGITUDE FROM  T_MASTER_CORD'),

        ('varlist', 'AIRPORT_ID,AIRPORT,DISPLAY_AIRPORT_NAME,DISPLAY_AIRPORT_CITY_NAME_FULL,LATITUDE,LONGITUDE'),

        ('grouplist', ''),

        ('suml', ''),

        ('sumRegion', ''),

        ('filter1', 'title='),

        ('filter2', 'title='),

        ('geo', 'Not Applicable'),

        ('time', 'Not Applicable'),

        ('timename', 'N/A'),

        ('GEOGRAPHY', 'All'),

        ('XYEAR', 'All'),

        ('FREQUENCY', 'All'),

        ('VarDesc', 'AirportSeqID'),

        ('VarDesc', 'AirportID'),

        ('VarDesc', 'Airport'),

        ('VarDesc', 'AirportName'),

        ('VarDesc', 'AirportCityName'),

        ('VarDesc', 'AirportWacSeqID2'),

        ('VarDesc', 'AirportWac'),

        ('VarDesc', 'AirportCountryName'),

        ('VarDesc', 'AirportCountryCodeISO'),

        ('VarDesc', 'AirportStateName'),

        ('VarDesc', 'AirportStateCode'),

        ('VarDesc', 'AirportStateFips'),

        ('VarDesc', 'CityMarketSeqID'),

        ('VarDesc', 'CityMarketID'),

        ('VarDesc', 'CityMarketName'),

        ('VarDesc', 'CityMarketWacSeqID2'),

        ('VarDesc', 'CityMarketWac'),

        ('VarDesc', 'LatDegrees'),

        ('VarDesc', 'LatHemisphere'),

        ('VarDesc', 'LatMinutes'),

        ('VarDesc', 'LatSeconds'),

        ('VarDesc', 'Latitude'),

        ('VarDesc', 'LonDegrees'),

        ('VarDesc', 'LonHemisphere'),

        ('VarDesc', 'LonMinutes'),

        ('VarDesc', 'LonSeconds'),

        ('VarDesc', 'Longitude'),

        ('VarDesc', 'UTCLocalTimeVariation'),

        ('VarDesc', 'AirportStartDate'),

        ('VarDesc', 'AirportEndDate'),

        ('VarDesc', 'AirportIsClosed'),

        ('VarDesc', 'AirportIsLatest'),

        ('VarType', 'Num'),

        ('VarType', 'Num'),

        ('VarType', 'Char'),

        ('VarType', 'Char'),

        ('VarType', 'Char'),

        ('VarType', 'Num'),

        ('VarType', 'Num'),

        ('VarType', 'Char'),

        ('VarType', 'Char'),

        ('VarType', 'Char'),

        ('VarType', 'Char'),

        ('VarType', 'Char'),

        ('VarType', 'Num'),

        ('VarType', 'Num'),

        ('VarType', 'Char'),

        ('VarType', 'Num'),

        ('VarType', 'Num'),

        ('VarType', 'Num'),

        ('VarType', 'Char'),

        ('VarType', 'Num'),

        ('VarType', 'Num'),

        ('VarType', 'Num'),

        ('VarType', 'Num'),

        ('VarType', 'Char'),

        ('VarType', 'Num'),

        ('VarType', 'Num'),

        ('VarType', 'Num'),

        ('VarType', 'Char'),

        ('VarType', 'Char'),

        ('VarType', 'Char'),

        ('VarType', 'Num'),

        ('VarType', 'Num'),

        ('VarName', 'AIRPORT_ID'),

        ('VarName', 'AIRPORT'),

        ('VarName', 'DISPLAY_AIRPORT_NAME'),

        ('VarName', 'DISPLAY_AIRPORT_CITY_NAME_FULL'),

        ('VarName', 'LATITUDE'),

        ('VarName', 'LONGITUDE'),

    ]



    r = requests.post('https://www.transtats.bts.gov/DownLoad_Table.asp', headers=headers, params=params, cookies=cookies, data=data)

    with open("data/airports.csv.zip") as f:

        f.write(r.content)





def download_timeseries(date):
    month_name = calendar.month_name[date.month]
    year = date.year
    month = date.month
    data = [
        ('UserTableName', 'On_Time_Performance'),
        ('DBShortName', 'On_Time'),
        ('RawDataTable', 'T_ONTIME'),
        ('sqlstr', ' SELECT FL_DATE,ORIGIN,CRS_DEP_TIME,DEP_TIME,CRS_ARR_TIME,ARR_TIME FROM  T_ONTIME WHERE Month ={} AND YEAR={}'.format(month, year)),
        ('varlist', 'FL_DATE,ORIGIN,CRS_DEP_TIME,DEP_TIME,CRS_ARR_TIME,ARR_TIME'),
        ('filter1', 'title='),
        ('filter2', 'title='),
        ('geo', 'All'),
        ('time',month_name),
        ('timename', 'Month'),
        ('GEOGRAPHY', 'All'),
        ('XYEAR', str(year)),
        ('FREQUENCY', '1'),
        ('VarDesc', 'Year'),
        ('VarType', 'Num'),
        ('VarDesc', 'Quarter'),
        ('VarType', 'Num'),
        ('VarDesc', 'Month'),
        ('VarType', 'Num'),
        ('VarDesc', 'DayofMonth'),
        ('VarType', 'Num'),
        ('VarDesc', 'DayOfWeek'),
        ('VarType', 'Num'),
        ('VarName', 'FL_DATE'),
        ('VarDesc', 'FlightDate'),
        ('VarType', 'Char'),
        ('VarDesc', 'UniqueCarrier'),
        ('VarType', 'Char'),
        ('VarDesc', 'AirlineID'),
        ('VarType', 'Num'),
        ('VarDesc', 'Carrier'),
        ('VarType', 'Char'),
        ('VarDesc', 'TailNum'),
        ('VarType', 'Char'),
        ('VarDesc', 'FlightNum'),
        ('VarType', 'Char'),
        ('VarDesc', 'OriginAirportID'),
        ('VarType', 'Num'),
        ('VarDesc', 'OriginAirportSeqID'),
        ('VarType', 'Num'),
        ('VarDesc', 'OriginCityMarketID'),
        ('VarType', 'Num'),
        ('VarName', 'ORIGIN'),
        ('VarDesc', 'Origin'),
        ('VarType', 'Char'),
        ('VarDesc', 'OriginCityName'),
        ('VarType', 'Char'),
        ('VarDesc', 'OriginState'),
        ('VarType', 'Char'),
        ('VarDesc', 'OriginStateFips'),
        ('VarType', 'Char'),
        ('VarDesc', 'OriginStateName'),
        ('VarType', 'Char'),
        ('VarDesc', 'OriginWac'),
        ('VarType', 'Num'),
        ('VarDesc', 'DestAirportID'),
        ('VarType', 'Num'),
        ('VarDesc', 'DestAirportSeqID'),
        ('VarType', 'Num'),
        ('VarDesc', 'DestCityMarketID'),
        ('VarType', 'Num'),
        ('VarDesc', 'Dest'),
        ('VarType', 'Char'),
        ('VarDesc', 'DestCityName'),
        ('VarType', 'Char'),
        ('VarDesc', 'DestState'),
        ('VarType', 'Char'),
        ('VarDesc', 'DestStateFips'),
        ('VarType', 'Char'),
        ('VarDesc', 'DestStateName'),
        ('VarType', 'Char'),
        ('VarDesc', 'DestWac'),
        ('VarType', 'Num'),
        ('VarName', 'CRS_DEP_TIME'),
        ('VarDesc', 'CRSDepTime'),
        ('VarType', 'Char'),
        ('VarName', 'DEP_TIME'),
        ('VarDesc', 'DepTime'),
        ('VarType', 'Char'),
        ('VarDesc', 'DepDelay'),
        ('VarType', 'Num'),
        ('VarDesc', 'DepDelayMinutes'),
        ('VarType', 'Num'),
        ('VarDesc', 'DepDel15'),
        ('VarType', 'Num'),
        ('VarDesc', 'DepartureDelayGroups'),
        ('VarType', 'Num'),
        ('VarDesc', 'DepTimeBlk'),
        ('VarType', 'Char'),
        ('VarDesc', 'TaxiOut'),
        ('VarType', 'Num'),
        ('VarDesc', 'WheelsOff'),
        ('VarType', 'Char'),
        ('VarDesc', 'WheelsOn'),
        ('VarType', 'Char'),
        ('VarDesc', 'TaxiIn'),
        ('VarType', 'Num'),
        ('VarName', 'CRS_ARR_TIME'),
        ('VarDesc', 'CRSArrTime'),
        ('VarType', 'Char'),
        ('VarName', 'ARR_TIME'),
        ('VarDesc', 'ArrTime'),
        ('VarType', 'Char'),
        ('VarDesc', 'ArrDelay'),
        ('VarType', 'Num'),
        ('VarDesc', 'ArrDelayMinutes'),
        ('VarType', 'Num'),
        ('VarDesc', 'ArrDel15'),
        ('VarType', 'Num'),
        ('VarDesc', 'ArrivalDelayGroups'),
        ('VarType', 'Num'),
        ('VarDesc', 'ArrTimeBlk'),
        ('VarType', 'Char'),
        ('VarDesc', 'Cancelled'),
        ('VarType', 'Num'),
        ('VarDesc', 'CancellationCode'),
        ('VarType', 'Char'),
        ('VarDesc', 'Diverted'),
        ('VarType', 'Num'),
        ('VarDesc', 'CRSElapsedTime'),
        ('VarType', 'Num'),
        ('VarDesc', 'ActualElapsedTime'),
        ('VarType', 'Num'),
        ('VarDesc', 'AirTime'),
        ('VarType', 'Num'),
        ('VarDesc', 'Flights'),
        ('VarType', 'Num'),
        ('VarDesc', 'Distance'),
        ('VarType', 'Num'),

        ('VarDesc', 'DistanceGroup'),

        ('VarType', 'Num'),

        ('VarDesc', 'CarrierDelay'),

        ('VarType', 'Num'),

        ('VarDesc', 'WeatherDelay'),

        ('VarType', 'Num'),

        ('VarDesc', 'NASDelay'),

        ('VarType', 'Num'),

        ('VarDesc', 'SecurityDelay'),

        ('VarType', 'Num'),

        ('VarDesc', 'LateAircraftDelay'),

        ('VarType', 'Num'),

        ('VarDesc', 'FirstDepTime'),

        ('VarType', 'Char'),

        ('VarDesc', 'TotalAddGTime'),

        ('VarType', 'Num'),

        ('VarDesc', 'LongestAddGTime'),

        ('VarType', 'Num'),

        ('VarDesc', 'DivAirportLandings'),

        ('VarType', 'Num'),

        ('VarDesc', 'DivReachedDest'),

        ('VarType', 'Num'),

        ('VarDesc', 'DivActualElapsedTime'),

        ('VarType', 'Num'),

        ('VarDesc', 'DivArrDelay'),

        ('VarType', 'Num'),

        ('VarDesc', 'DivDistance'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div1Airport'),

        ('VarType', 'Char'),

        ('VarDesc', 'Div1AirportID'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div1AirportSeqID'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div1WheelsOn'),

        ('VarType', 'Char'),

        ('VarDesc', 'Div1TotalGTime'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div1LongestGTime'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div1WheelsOff'),

        ('VarType', 'Char'),

        ('VarDesc', 'Div1TailNum'),

        ('VarType', 'Char'),

        ('VarDesc', 'Div2Airport'),

        ('VarType', 'Char'),

        ('VarDesc', 'Div2AirportID'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div2AirportSeqID'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div2WheelsOn'),

        ('VarType', 'Char'),

        ('VarDesc', 'Div2TotalGTime'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div2LongestGTime'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div2WheelsOff'),

        ('VarType', 'Char'),

        ('VarDesc', 'Div2TailNum'),

        ('VarType', 'Char'),

        ('VarDesc', 'Div3Airport'),

        ('VarType', 'Char'),

        ('VarDesc', 'Div3AirportID'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div3AirportSeqID'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div3WheelsOn'),

        ('VarType', 'Char'),

        ('VarDesc', 'Div3TotalGTime'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div3LongestGTime'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div3WheelsOff'),

        ('VarType', 'Char'),

        ('VarDesc', 'Div3TailNum'),

        ('VarType', 'Char'),

        ('VarDesc', 'Div4Airport'),

        ('VarType', 'Char'),

        ('VarDesc', 'Div4AirportID'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div4AirportSeqID'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div4WheelsOn'),

        ('VarType', 'Char'),

        ('VarDesc', 'Div4TotalGTime'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div4LongestGTime'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div4WheelsOff'),

        ('VarType', 'Char'),

        ('VarDesc', 'Div4TailNum'),

        ('VarType', 'Char'),

        ('VarDesc', 'Div5Airport'),

        ('VarType', 'Char'),

        ('VarDesc', 'Div5AirportID'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div5AirportSeqID'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div5WheelsOn'),

        ('VarType', 'Char'),

        ('VarDesc', 'Div5TotalGTime'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div5LongestGTime'),

        ('VarType', 'Num'),

        ('VarDesc', 'Div5WheelsOff'),

        ('VarType', 'Char'),

        ('VarDesc', 'Div5TailNum'),

        ('VarType', 'Char')

    ]

    cookies = {

        'ASPSESSIONIDCAQCSDSS': 'GOBIGIDBLGILGICKLFHKIHMN',

        '__utmt_ritaTracker': '1',

        '__utmt_GSA_CP': '1',

        '__utma': '261918792.554646962.1504352085.1504442392.1504442407.3',

        '__utmb': '261918792.8.10.1504442407',

        '__utmc': '261918792',

        '__utmz': '261918792.1504442407.3.2.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided)',

    }



    headers = {

        'Origin': 'https://www.transtats.bts.gov',

        'Accept-Encoding': 'gzip, deflate, br',

        'Accept-Language': 'en-US,en;q=0.8',

        'Upgrade-Insecure-Requests': '1',

        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.101 Safari/537.36',

        'Content-Type': 'application/x-www-form-urlencoded',

        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',

        'Cache-Control': 'max-age=0',

        'Referer': 'https://www.transtats.bts.gov/DL_SelectFields.asp',

        'Connection': 'keep-alive',

    }



    params = (

        ('Table_ID', '236'),

        ('Has_Group', '3'),

        ('Is_Zipped', '0'),

    )



    r = requests.get('https://www.transtats.bts.gov/DownLoad_Table.asp',

                     headers=headers, params=params,

                     cookies=cookies, data=data)

    with open("data/timeseries/{:%Y-%m}.zip".format(date.to_timestamp()), "wb") as f:

        f.write(r.content)