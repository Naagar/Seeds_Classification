import dash
import dash_html_components as html
import random

app = dash.Dash(assets_folder='./assets')
app.layout = html.Div(  
                        children=[html.Img(className='icon')] , 
                        # style={"height": "10px", "width": "19px"}
                    )






if __name__ == '__main__':
    port = random.randrange(2000, 7999)
    # during development
    # app.run_server(host='127.0.0.1', port=port, debug=True)


    # for testing
    app.run_server(host='127.0.0.1', port=port, debug=True)

