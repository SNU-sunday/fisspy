from __future__ import absolute_import
from astropy.utils.data import download_file
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as bs
from pandas import DataFrame
from os.path import join
from shutil import move

url = 'http://fiss.snu.ac.kr/static/data/'

def search(year, month=None, day=None):
    """
    search

    Search a zip file of the FISS compressed data.

    Parameters
    ----------
    year: `int` or `str`
        A year to query data. If year = 'all', search all data
    month: `int` (optional)
        A month to query data.
    day: `int` (optional)
        A day to query data.

    Returns
    -------
    search_res: `~pandas.core.frame.DataFrame`
        Table of searching results.

    Example
    -------
    >>> from fisspy.data import search
    >>> res = search(2014, 6, 3)
    """

    req = Request('http://fiss.snu.ac.kr/data_list/')
    res = urlopen(req)
    rr = res.read()
    bb = bs(rr, 'html.parser')
    aL = bb.findAll('a')
    linkL = []
    dateL = []
    targetL = []
    for a in aL:
        link = a.attrs['href']
        linkL += [link]
        split_link = link.split('/')
        date = split_link[6]
        y = date[:4]
        m = date[4:6]
        d = date[6:8]
        date = f'{y}-{m}-{d}'
        target = split_link[-1][:-4]
        try:
            if int(year) == int(y):
                if month is not None and day is not None:
                    if int(month) == int(m) and int(day) == int(d):
                        dateL += [date]
                        targetL += [target]
                elif month is not None and day is None:
                    if int(month) == int(m):
                        dateL += [date]
                        targetL += [target]
                elif month is None and day is None:
                    dateL += [date]
                    targetL += [target]
        except:
            if year == 'all':
                dateL += [date]
                targetL += [target]

    res_dict = {'Date': dateL,
                'Target': targetL}
    search_res = DataFrame(res_dict)
    return search_res

def download(search_res, save_path):
    """
    download

    Download the FISS compressed data.

    Parameters
    ----------
    search_res: `~pandas.core.frame.DataFrame`
        Table of searching results.
    save_path: `str`
        Directory path to save the downloaded data.

    Returns
    -------
    None

    Example
    -------
    >>> from fisspy.data import download
    >>> download(res, r'D:\test')
    """


    n = search_res.index.stop - search_res.index.start
    for ii, i in enumerate(search_res.index):
        y, m, d = search_res.Date[i].split('-')
        fname = join(save_path, f'{y}{m}{d}_{search_res.Target[i]}.zip')
        durl = f'{url}{y}/{y}{m}{d}/data/{search_res.Target[i]}.zip'

        print(f'Downloading {ii+1}/{n}')
        df = download_file(durl)
        move(df, fname)
    print('Done')
