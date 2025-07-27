import React from 'react'
import Testimonialinfo from './testimonialinfo';
import { VscVerifiedFilled } from "react-icons/vsc";
import { RxSketchLogo } from "react-icons/rx";

const AutoSlider = () =>{
   return(
      <div className='entireBlog'>
      <div className="Apna_Blog">
             <h3>Testimonials</h3>
       </div>
       <div className='OnCEpure'>
       {Testimonialinfo.map((test)=>{
        return (
        <>
       <div className='TestimonialCard' key={test.id}>
       <div className='Testimonial_top'>
           <img src= {test.img} className='Testimonial_Image'></img>
           <div className='Testimonial_User'><h2>{test.user}</h2></div>
           </div>
           <hr />
           <div className='Test_btm'>
           <div className='Veruser'><h4>Verified user <VscVerifiedFilled /></h4></div>
           <div className='Testimonial_content'><p>{test.content}</p></div>
           </div>
       </div>
       </>
        )
       })}
      
 </div>
</div>
);
   }
export default AutoSlider;