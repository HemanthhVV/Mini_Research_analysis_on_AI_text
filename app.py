from fasthtml.common import *
import pickle
import helpers
from decimal import Decimal

css = Style(':root {--pico-font-size:90%,--pico-font-family: Pacifico, cursive;}')
app = FastHTMLWithLiveReload(hdrs=(picolink,css))


class Statsmodel:
    statsmodel = None
    def __init__(self) -> pickle:
        self.statsmodel = pickle.load(open('./statsmodel.sav','rb'))

class Vectormodel:
    vectormodel = None
    def __init__(self) -> pickle:
        self.vectormodel = pickle.load(open('./vectorizermodel.sav','rb'))

class Vectorizer:
    vectorizer = None
    def __init__(self) -> pickle:
        self.vectorizer = pickle.load(open('./vectorizer.sav','rb'))


stats_model = Statsmodel()
vector_model = Vectormodel()
vect = Vectorizer()

def textarea(**kw):
    return Textarea(name="text_input", placeholder="Enter your text here...", rows="10", cls="text-area",**kw)

def convert_to_number(probs:list)->list:
    # num1 = Decimal(float(probs[0]))
    # num2 = Decimal(float(probs[1]))
    # return [f"{num1:.4f}",f"{num2:.4f}"]
    probs = list(probs)
    num1 = round(float(probs[0])*100,3)
    print(num1)
    num2 = round(float(probs[1])*100,3)
    print(num2)
    return [str(num1),str(num2)]

@app.get("/")
def welcome(content:list|None=None):
    space = Div(style="height:40px")
    form = Form(
        textarea(),
        space,
        Button("Submit", type="submit", cls="btn"),
        action="/", method="post"
    )
    div1,div2,div3 = None,None,None
    div1_contents = convert_to_number(content[0][0]) if content else None
    div2_contents = convert_to_number(content[1][0]) if content else None
    div3_contents = convert_to_number(content[2]) if content else None
    heading = Div(H1("Check your Text here"),style="display: flex;justify-content: center; ")
    if div1_contents:
        div1 = Div(Card(Div(H2("On Statistics model",style="display: flex;justify-content: center;")),H3(div1_contents[0] + "% LLM",style="color:lightcoral"),H3(div1_contents[1] + "% Human",style="color:green"),style="display: flex;align-items: center"))
    if div2_contents:
        div2 = Div(Card(Div(H2("On Vect model",style="display: flex;justify-content: center;")),H3(div2_contents[0] + "% LLM",style="color:lightcoral"),H3(div2_contents[1] + "% Human",style="color:green"),style="display: flex;align-items: center"))
    if div3_contents:
        div3 = Div(Card(Div(H2("On RoBERT-a model",style="display: flex;justify-content: center;")),H3(div3_contents[0] + "% LLM",style="color:lightcoral"),H3(div3_contents[1] + "% Human",style="color:green"),style="display: flex;align-items: center"))

    if any([div1,div2,div3]):
        return Title("Detect Text"), Main(heading,space,Card(form),space,Grid(div1,div2,div3),cls='container')
    else:
        return Title("Detect Text"), Main(heading,space,Card(form),cls='container')



@app.post('/')
async def saving(text_input:str):
    form_data = text_input
    content = [stats_model.statsmodel.predict_proba(helpers.makeTestingDataStats(text_input)),
               vector_model.vectormodel.predict_proba(helpers.makeTestingDataVect(text_input)),
               helpers.classify_user_input(text_input)]
    return welcome(content=content)

serve()
