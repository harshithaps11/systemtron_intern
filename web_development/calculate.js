let input = document.getElementById('inputBox');

let string = "";
$(".button").on('click', (e) =>{
    if(e.target.innerHTML == '='){
        string = eval(string);  // evaluate string 2+7 ,return 9
        input.value = string;
    }
    else if(e.target.innerHTML == 'AC'){
        string = "";
        input.value = string;
    }
    else if(e.target.innerHTML == 'DEL'){
        string = string.substring(0, string.length-1);
        input.value = string;
    }
    else{
        console.log(e);
        console.log(e.target)
        string += e.target.innerHTML;
        input.value = string;
    }
     
});