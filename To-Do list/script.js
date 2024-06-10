// Function called while clicking add button
function add_item() {

    // Getting box and ul by selecting id;
    let itemInBox = document.getElementById("box");
    let list_item = document.getElementById("list_item");
    if(itemInBox.value != ""){
    
        // Creating element and adding value to it
        let make_li = document.createElement("LI");
        make_li.appendChild(document.createTextNode(itemInBox.value));
    
        // Adding li to ul
        list_item.appendChild(make_li);
    
        // Reset the value of box
        itemInBox.value=""
        
        // Delete a li item on click
        make_li.onclick = function(){
            this.parentNode.removeChild(this);
        }
    
    }
    else{
        // Alert msg when value of box is "" empty.
        alert("plz add a value to item");
    }
    
    }
//this is a comment