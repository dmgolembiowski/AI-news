$(document).ready(function() {
  
    
  ///the button to What
  $('div.seeWhat').on("click",function() {
    window.location.href = link_what; 
  }); 


    
  $("section#intro").addClass('loaded'); 



    /*  instagram section */
    
    $("figcaption .toggle_icon").on("click", function(){
        var thisCap = $(this).parents("figcaption");

        if(thisCap.hasClass("active")){
            $(this).transition({rotate: '0deg'},500).removeClass("rotated");
            thisCap.removeClass("active", 600);
            thisCap.find(".toggle").slideUp();
        }else{
          $(this).transition({rotate: '45deg'},500).addClass("rotated");
            thisCap.addClass("active", 600);
            thisCap.find(".toggle").slideDown();
        }

    });//END of toggle_icon

    function enableIGCarousel(){
        //adds class for monbile experience
        $(".journalist_ig").addClass("carousel_ig");

        //IG slider
        var igSlider = $('.journalist_ig').bxSlider({
            infiniteLoop: false
        });

    }//END of enableIGCarousel

    if(WURFL.form_factor === "Smartphone"){
        console.log("its a phone");
        enableIGCarousel();//set up the mobile slider for the IG boxes

    }

   /* END of instagram section */

   //-----------------------
    
   // GLOBE / LOCATIONS //

     var globeInUse = 1; //might be 3 or more globe images

     function goFind(target) {
        //chagne the globe if need be, and show the journalists in the target location
        var topper = '40%'; 
        var lefter = '40%'; 
        switch(target) {
     
        case("Washington"):
          var globeToUse = '1'; 

          break;
          case("Beijing"):
          var globeToUse = '6'; 

          
          break;
          case("Lapaz"):
          var globeToUse = '3'; 
         
          break;
          case("MexicoCity"):
          var globeToUse = '2'; 

          break;
          case("Paris"):
          var globeToUse = '4'; 
         
          break;

          case("Jerusalem"):
          var globeToUse = '5'; 
      
          break;

          case("Sydney"):
          var globeToUse = '7'; 
         
          break;
        }


        var wdth = window.innerWidth; 
        if(wdth<900) {
          topper = topper - 10; 
        }

        if(wdth<600) {
          //lefter = 10; 
        }

        topper += '%'; 
        lefter += '%'; 


        $('location.active').removeClass('active').fadeOut(); 

        if(globeToUse!=globeInUse) {
          //need to change the globe image...
          console.log("need to change Earth"); 
          //var glober = 'images/earth/globe'+globeToUse+'.png'; 

          // Show the one on-deck
          $('#bg_globe'+globeToUse).show(); //it's behind bc zindex .. 

          //fade out the globe
          $('.bg_globe.activeGlobe').fadeOut('2000',function() {
            $(this).removeClass('activeGlobe').removeClass('hiZ'); 
            console.log("the active globe is faded out");  //globe is faded out.. 

            //the one in Waiting has to be shown .. 
            $('#bg_globe'+globeToUse).addClass('activeGlobe').addClass('hiZ'); //
            
            //show the target Location 
             showJournalist(target); 

          });

          globeInUse = globeToUse; 

        } else { //using the Same Globe 
            
             //show the target Location 
             
         showJournalist(target); 
            
    

        }

        //set the Top / Left position of Location 
         
     function showJournalist(target) {
      
      $('#earth').find('#'+target).fadeIn().addClass('active'); 

     }

     } //end of function goFind 

  


        $('locLink').on("click",function() {
          $('locLink.active').removeClass('active'); //remove existing active link
          $(this).addClass('active');  //give selector 'active' to the Link 
          var target = $(this).data('target'); //find the target
          
          console.log("target is "+target); 
          
       
          goFind(target); 
        }); 

     


    goFind("Washington");     
    
    
    //END of Globe , Earth , on the ground Section
    
    //-----------------------
 
    //-----------------------

  

    //---------------------


    $('vidLink').on("click",function() {

      $('vidLink.active').removeClass('active'); 
      $(this).addClass('active'); 
      var target = $(this).data('target'); 
      console.log("target is "+target)

       makeYoutube(target); 
    }); 

    function makeYoutube(subject) {
        var vid = $('#nyt_bl_video'); 
        //alert("make yt for "+subject)
        switch(subject) {
            case("hicks"):
            yt_src = 'zs4-rb0f7HI'; 
            break;
            case("berehulak"):
            yt_src = '4_3PtE7358c'; 
            break;
            case("denton"):
            yt_src = '5r1mG239wRM'; 
            break;
            case("mazzetti"):
            yt_src = '-BhqE2nSDlc'; 
            break;
            case("haner"):
            yt_src = 'fmqC_Ufaa2Y'; 
            break;
        }

        newsrc = 'https://www.youtube.com/embed/'+yt_src+'?enablejsapi=1&rel=0&modestbranding=1&autohide=1&showinfo=0&controls=0'; 
        vid.attr("src",newsrc);

    } //end of make youtube 

    makeYoutube('hicks');

  //end of videos //

  //our Methods
  //append trump link
  var methodLink = "https://www.nytimes.com/2017/10/05/us/harvey-weinstein-harassment-allegations.html";    
  $('.method-link').attr({href: methodLink, target: "_blank"});
  
  var trumpLink = "https://www.nytimes.com/2016/10/02/us/politics/donald-trump-taxes.html";    
  $('#trumpTaxLink').attr({href: trumpLink, target: "_blank"});    

}); // end of doc ready

var ytTag = document.createElement('script');
ytTag.src = "https://www.youtube.com/iframe_api";
var firstScriptTag = document.getElementsByTagName('script')[0];
firstScriptTag.parentNode.insertBefore(ytTag, firstScriptTag);

var ytPlayer;
function onYouTubeIframeAPIReady() {
    ytPlayer = new YT.Player( 'nyt_bl_video', {
        videoId: 'zs4-rb0f7HI',
        events: {
            'onStateChange': onPlayerStateChange
        }
    });
}

function onPlayerStateChange(event) {
    if(event.data == YT.PlayerState.PLAYING){
        newsrc = 'https://www.youtube.com/embed/'+yt_src+'?enablejsapi=1&autoplay=1';
        $('#nyt_bl_video').attr("src",newsrc);
    }  
}


