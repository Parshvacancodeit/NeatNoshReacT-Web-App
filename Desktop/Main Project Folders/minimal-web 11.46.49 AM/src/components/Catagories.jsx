import { useState } from "react";
import CatCardLayout from "./CatCardLayout";
import CatagoriesCardInfo from "../CatagoriesCardInfo";
import React from 'react'

const Catagories =()=>{
    const [data,setdata]=useState(CatagoriesCardInfo);
    const filterresult=(catItems)=>{
        const result = CatagoriesCardInfo.filter((curdata)=>{
            return curdata.category===catItems;
        })
        setdata(result);
    }
    return (
        <div className="Catagories_sec">
        <h3 className="catagorSec">Search By Catagories</h3>
        <hr id="breakLine" />

        <ul>
            <button onClick={()=>
    filterresult('Cups')
}>Cups</button>
            <button onClick={()=>
    filterresult('Cups')
}>Dinnerware Sets</button>
            <button onClick={()=>
    filterresult('Cups')
}>Serving Sets</button>
          <button onClick={()=>
    filterresult('Cups')
}>Tea Sets</button>
          <button onClick={()=>
    filterresult('Cups')
}>Limited Edition</button>
            </ul>
        </div>
    );
}

export default Catagories;

