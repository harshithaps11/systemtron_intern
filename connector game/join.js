var playerRed="R";
var playerYellow="Y";
var currPlayer='playerRed';

var gameOver= false;
var board;

var rows=6;
var columns=7;

window.onload = function() {
    setGame();
}

function setGame() {
    board = [];

    for (let r=0; r<rows; r++){
        let row = [];
        for (let c=0; c<columns; c++){
            row.push(' ');
            //<div id='0-0' class='tile'></div>
            let tile = document.createElement("div");
            tile.id = r.toString()+'-'+c.toString();
            tile.classList.add('tile');
            tile.addEventListener("click,setPiece")
            document.getElementById("board").append(tile);
        }
        board.push(row);

    }
}

function setPiece(){
    if (gameOver){
        return;
    }
    let coords = this.id.split("-");//"0-0"->["0","0"]
    let r= parseInt(coords[0]);
    let c = parseInt(coords[1]);

    board[r][c] = currPlayer;
    let tile = this;
    if (currPlayer == playerRed){
        tile.classList.add("red-piece");
    }
    else{
        tile.classList.add("yellow-piece")
    }

}
