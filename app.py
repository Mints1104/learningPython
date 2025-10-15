from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return """
<h1>Hello,World!</h1>"
<p>Go to a user's porofile:</p>
<ul>
<li><a href="/profile/Harun">Harun's Profile</a></li>
<li><a href="/profile/Alice">Alice's Profile</a></li>

"""
@app.route("/profile/<username>")
def profile(username):
    return f"<h1>Helllo, {username}! This is your profile page.</h1>"

if __name__ == '__main__':
    app.run(debug=True)