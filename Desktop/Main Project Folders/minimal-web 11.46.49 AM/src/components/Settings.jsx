import { MdAccountCircle } from "react-icons/md";
import Settingsdata from "./Settingsdata";
const Settings = () => {
    return(
        <>
          <div className="EntireSettbox">
          <div className="Accountinfo">
             <div className="Contdiv">
               <h3>This is a div</h3>
             </div>
          </div>
            <div className="Contentsabz">
             {Settingsdata.map((values)=>
                  <div className="Contincont" key={values.id}>
                    <p>{values.name}</p>
                  </div>
                  )}
            </div>
          </div>
        </>
    );
}

export default Settings;