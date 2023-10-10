

__author__ = "Juhyung Kang"
__email__ = "jhkang@astro.snu.ac.kr"


def nbhtml2html(nbhtml):
    r = open(nbhtml, encoding='UTF8').readlines()
    html = ''.join(r)
    argE1 = html.find('* IPython base') + 20
    html= html[argE1:]
    tmp = html.find('<style')
    
    argE2 = html.find('<style', tmp+5)
    c1 = html[:argE2]
    argE3 = html.find('<link')
    c2 = html[argE3:]
    html = c1 + c2
    header = """{% extends 'template.html' %}
{% set active_page = "Guide" %}

{% block content %}
<style type="text/css">\n\n"""
    html = header + html
    argE4 = html.find('</body>') + 9
    footer = "{% endblock %}"
    html = html[:argE4] + footer
    
    of = open(nbhtml, 'w')
    of.write(html)
    of.close()

def md2html(file):
    None

if __name__ == '__main__':
    f = 'C:/Users/dlooi/Downloads/fisspy.html'
    nbhtml2html(f)
    

    
    