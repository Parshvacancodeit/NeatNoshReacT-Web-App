import React from "react";
import ReactDOM from "react-dom";
import  NewCardInfo from "../NewCardInfo";
import NewLaunchCard from "./NewLaunchCard";
import { Link } from "react-router-dom";

const Card = () => { 
    return (
     NewCardInfo.map((e)=>{
       return (
        <Link style={{textDecoration: 'none',color:"black"}} to={`/productview/${e.id}`}>
       <NewLaunchCard key={e.id} 
    
       image={e.image}
       name={e.name}
       desc={e.desc}
       price={e.price}
       link={e.link}
       >
     </NewLaunchCard>
    </Link>
     );})
  
  );

       } 

export default  Card;