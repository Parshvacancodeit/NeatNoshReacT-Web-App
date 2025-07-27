import React from "react";
import { Link } from "react-router-dom";

const NewLaunchCard = (props) =>{
    return(
     
        <div className="NewMainCard" key={props.id}>
        
          <img className="NewMainImage" src={props.image}/>
          <hr/>
          <div className="NewCardBody">
            <h2 className="NewMainTitle">{props.name}</h2>
            <p className="NewMainDesc">{props.desc}</p>
            <h4 className="NewMainPrice">{props.price}</h4>
            <button className="NewMainCheck" src={props.link}>Check It Out</button>
            </div>
         
        </div>
      

  
    );
}
export default NewLaunchCard;
