import React, { useEffect, useRef, useState } from "react";

const Timer = ({duration})=>{
    const [Time,SetTime] = useState(duration)
    
    useEffect(()=>{
       setTimeout (()=>{
          SetTime(Time - 1000);
       },1000)

    },[Time]);
    const getFormattedtime= (millisecond) => {
      let total_sec = parseInt(Math.floor(millisecond/1000));
      let total_min = parseInt(Math.floor(total_sec/60));
      let total_hr =parseInt(Math.floor(total_min/60));
      let days =parseInt(Math.floor(total_hr/24));

      let second = parseInt(total_sec % 60);
      let minutes =parseInt(total_min % 60);
      let hours = parseInt(total_hr % 24 );
      
      return(
        `  ${days}:${hours}:${minutes}:${second}`
      );
    };
   return(
     <div className="CollapseTime">{getFormattedtime(Time)}</div>
   );
}
export default Timer;