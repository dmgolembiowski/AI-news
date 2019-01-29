/// GENERAL JS /// 
// USED BY BOTH WHY AND WHAT //
//CALLED before UNIQUE JS (why, what)


//header and footer //

//header
function getParameterByName(name)
{
    name = name.replace(/[\[]/, "\\\[").replace(/[\]]/, "\\\]");
    var regexS = "[\\?&]" + name + "=([^&#]*)";
    var regex = new RegExp(regexS);
    var results = regex.exec(window.location.search);
    if(results == null)
        return "";
    else
        return decodeURIComponent(results[1].replace(/\+/g, " "));
}

var cid = getParameterByName("campaignId");

var link_cta = 'https://www.nytimes.com/subscriptions/Multiproduct/lp8HYKU.html?campaignId='+cid; 

/// NAVIGATION //// 
function makeLinks() {
    
    
    var encodeLp = getParameterByName("lp") || "";
    var encodedlpSplit = encodeLp.split('?')[0];
    var encodedlp = encodeURIComponent(encodedlpSplit);
    var referLp = encodedlpSplit;
    // !!! Important, sanitizing !!! 
    if (!referLp.match(/^https?:\/\/www(\.stg|\.dev)?\.nytimes\.com/gi)){
        referLp = "";
        encodeLp = "";
    }
    
    //console.log(encodedlp);
    
    //alert(encodeLp);
    
    //alert(encodedlp);
	/*
	
	*/
   
    var link_why = 'lp8FW9H.html?campaignId='+cid; 
    var link_what = 'lp3L3W6.html?campaignId='+cid; 
    //certain links .. might need to dynamically add cid.. prob better to hard code it. 
    var link_hd ="https://www.nytimes.com/subscription/hd/1041.html?campaignId="+cid
    var link_newreader = 'https://www.nytimes.com/subscriptions/Multiproduct/lp8HYKU.html?campaignId='+cid; 
    var link_education = 'https://www.nytimes.com/subscriptions/edu/lp8LQFK.html?campaignId=6RR8R'; 
    var link_spotify = 'https://www.nytimes.com/subscriptions/Multiproduct/lp8U939.html?campaignId=6RQWY'; 
    var link_corporate = 'http://www.nytimescorporate.com/?Pardot_Campaign_Code_Form_Input=6RJUJ'; 
    var link_crosswords = 'https://www.nytimes.com/subscriptions/games/lp897H9.html?campaignId=6RRWR'; 
	var link_gift = 'https://www.nytimes.com/subscriptions/Multiproduct/lp8R34X.html?campaignId=3FQ4Y';
    
    
    if (referLp){
        //alert("yes we need refe page");
        referLp+='?campaignId='+cid;
        // applying already sanitized referrer lp URL
        //$('.link_cta,.sub_cta').attr('href',referLp);
        link_cta = referLp; 
        link_why = 'lp8FW9H.html?campaignId='+cid+'&lp='+encodedlp; 
        link_what = 'lp3L3W6.html?campaignId='+cid+'&lp='+encodedlp; 
        //certain links .. might need to dynamically add cid.. prob better to hard code it. 
        link_newreader = 'https://www.nytimes.com/subscriptions/Multiproduct/lp8HYKU.html?campaignId='+cid+'&lp='+encodedlp; 
        link_spotify = 'https://www.nytimes.com/subscriptions/Multiproduct/lp8U939.html?campaignId=6RQWY&lp='+encodedlp; 
    }
    
    $('.link_education').attr('href',link_education); 
    $('.link_newreader').attr('href',link_newreader); 
    //$('.link_spotify, #listenin .discover').attr('href',link_spotify); 
    $('.link_crosswords').attr('href',link_crosswords); 
    $('.link_corporate').attr('href',link_corporate); 
	$('.link_gift').attr('href',link_gift);
    $('.link_cta, .sub_cta, .subscribeNow').attr('href',link_cta);
    $('.link_why').attr('href',link_why); 
    $('.link_what').attr('href',link_what);
    $('.cta-hd').attr('href',link_hd);


        ///so.. some links open in new window.
    $('footer a,a.external').attr('target','_blank'); 
    
    
    
    
}



        function enableFixedNav(){
            
            var aboveAccessDevice = $("body > section").eq(1).offset().top - 100;
            var scrolledABit = $(".nav_bar").offset().top + 400;
            $(window).scroll(function(){
                var top = window.pageYOffset; 
                
                if(top >= scrolledABit){
                    
                    //check if for mobile, the menu slider, to reset it to close when making mobile nav fixed

                    //srolls alittle bit down (not to the next section), just enough not to see the absolute bar to just set the nav fixed but positioned slighty above the screen not to be visible
                    $(".nav_bar").addClass("fixed_nav");
                    
                    if(top >= aboveAccessDevice){
                        //we slide the fixed nav down to the intended position  giving it the nice slide in from top effect
                        $(".nav_bar.fixed_nav").addClass("slide_down", 300); 
                    }else{
                        //returns to the above position but still fixed
                        $(".nav_bar.fixed_nav").removeClass("slide_down", 200);
                    }   
  
                }else{
                    //resets the entire nav to be absolute to the top
                    $(".nav_bar").removeClass("fixed_nav");
                }

            });  
        }//END of enableFixedNav

        function enableMenu(){
            $(".mobile_nav_links h5").on("click", function(){
 
                if($("nav.nav_bar").hasClass("opened")){
                    $(".mobile_nav_links h5 i").transition({rotate: "0deg"});
                    $("nav.nav_bar").removeClass("opened", 500);
                    $(".mobile_nav_links menu").slideUp(500);
                }else{
                    $(".mobile_nav_links h5 i").transition({rotate: "180deg"});
                    $("nav.nav_bar").addClass("opened", 500);
                    $(".mobile_nav_links menu").slideDown(800);
                }
                
            });//END of mobile_nav_links h5 click

            if(thisPage==='why') { 
                $('menu .link_why').remove(); 
                $('.nav_links.desker a.link_why').addClass("activepage"); 

            } else { //what page 

                $('menu .link_what').remove();    
                $('.nav_links.desker a.link_what').addClass("activepage"); 
        
            }
            
        }//END of enableMenu
        


//footer 

function enableFooterAccord(){
    $("footer.footer .col h3 button.footer_toggle").on("click", function(){
        var theseLinks = $(this).parents(".col").find(".footer_links");
        var thisBtn =$(this).parents("h3");
        var thisIcon = $(this).parents("h3").find("i");
        var thisSvg = $(this).find("i svg g");
        
        if(thisBtn.hasClass("active")){
            thisBtn.removeClass("active");
            theseLinks.slideUp();
            thisIcon.transition({rotate:"0"});
        }else{
            thisBtn.addClass("active");
            theseLinks.slideDown();
            thisIcon.transition({rotate:"45"});
        }

    });//END of mobile_nav_links h5 click
    
}//END of enableMenu


function makeNav() {
  var placePut = $('.nav_bar'); 
  var url_nav = 'assets_pa2017/html/nav1_5.html'; 
  $.get(url_nav, function(connie) {

        placePut.html(connie); 
        enableMenu();
        enableFixedNav();
        makeLinks(); 
        //assure the Links will work.. call to setLinks
  }); 

}

function makeNavSub() {
  //display the logo .. 
   //will be handled via CSS body.why_subscribe  

}

function makeFooter() {
  var placePut = $('footer'); 
  var url_nav = 'assets_pa2017/html/footer.html'; 
  $.get(url_nav, function(connie) {

        placePut.html(connie); 
        enableFooterAccord();
        makeLinks(); 
        //assure the Links will work.. call to setLinks
  }); 

}



//end of header and footer 


//is mobile?

//give width and height 

//is Lando? is Porto ?

// these have to functions so can be Living / called at different times... 


//when doc ready 
$(document).ready(function() {
    
    var spaceForNav = 0; //extra space for fixed nav not to overlap the section 
    
    if(thisPage!='why_share') {
      makeNav(); 
      //makeNavSub();

      spaceForNav = 50;
         
    } 
	
    if (WURFL.form_factor === 'Smartphone') {
        $('body').addClass('mobile'); 

    }else if(WURFL.form_factor === 'Tablet'){

    }

    ///the Ready to Subscribe button is not an A tag .. so 
    $('div.subscribeNow').on("click",function() {
      window.location.href = link_cta; 
    });
    
    $("a.down_arrow").on("click", function(){  
        var nextSection  = $(this).parents("section").next("section").attr("id");
        
        $('html, body').animate({
            scrollTop: $("section#"+nextSection).offset().top - spaceForNav
        }, 800);       
    });
    

    makeFooter(); 

}); // end of doc ready 