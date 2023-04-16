from flask import Flask, render_template, request
import algorithme

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ingredients = request.form['ingredients']
        
        # Ici, on appelle l'algorithme qui traite les informations
        # et qui renvoie une table PrettyTable
        table = algorithme.all(ingredients)
        
        return render_template('index.html', table=table)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
