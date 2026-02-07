from flask import Flask,render_template,request,session,redirect,url_for
from app.components.retriever import create_qa_chain
from dotenv import load_dotenv
import os

from app.common.logger import get_logger

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

app = Flask(__name__)
app.secret_key = os.urandom(24)

logger = get_logger(__name__)

from markupsafe import Markup
def nl2br(value):
    return Markup(value.replace("\n" , "<br>\n"))

app.jinja_env.filters['nl2br'] = nl2br

@app.route("/" , methods=["GET","POST"])
def index():
    if "messages" not in session:
        session["messages"]=[]

    if request.method=="POST":
        user_input = request.form.get("prompt")

        if user_input:
            messages = session["messages"]
            messages.append({"role" : "user" , "content":user_input})
            session["messages"] = messages
            logger.info(f"User input received: {user_input}")
            try:
                qa_chain = create_qa_chain()
                logger.info(f"QAChain type: {type(qa_chain)}")
                logger.info(f"Input keys: {qa_chain.input_keys}")
                logger.info(f"Output keys: {qa_chain.output_keys}")

                response = qa_chain.invoke({"query" : user_input})
                logger.info(f"LLM response: {response}")
                result = response.get("result" , "No response")

                messages.append({"role" : "assistant" , "content" : result})
                session["messages"] = messages

            except Exception as e:
                logger.exception("Error while running QA chain")
                error_msg = f"Error: {repr(e)}"
                return render_template("index.html", messages=session["messages"], error=error_msg)

            
        return redirect(url_for("index"))
    return render_template("index.html" , messages=session.get("messages" , []))

@app.route("/clear")
def clear():
    session.pop("messages" , None)
    return redirect(url_for("index"))

if __name__=="__main__":
    app.run(host="0.0.0.0" , port=5000 , debug=False , use_reloader = False)



