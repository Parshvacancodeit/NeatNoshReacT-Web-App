import Timer from "./Timer";
const SalePage = () =>{
    return(
        <div className="WholeCardTime">
     <div className="SaleCardcss">
        <h1 className="Collapse2">Santa's Sleigh of Savings Ends In : </h1>
        </div>
        <div className="TimeCollapse"><h1><Timer duration={ 2*24*60*60*1000}/></h1>
        </div>
     </div>
    );
}

export default SalePage;