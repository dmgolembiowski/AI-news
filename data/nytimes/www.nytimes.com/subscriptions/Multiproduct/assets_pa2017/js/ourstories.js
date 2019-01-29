   $(document).ready(function() {
      
   
    /* our stories section */
    link_biles = 'https://www.nytimes.com/interactive/2016/10/24/world/asia/living-in-chinas-expanding-deserts.html'; 
    $('a#title_biles').attr({href:link_biles,target:'_blank'}); 



    var ourstories_vid = document.getElementById('ourstories_video'); 
    
    $("#ourstories .learnmore").on("click", function(){
            
            var prevBtnText = "See More Visual Stories";
            var thisIcon = $(this).find("i");
                        
            $("#ourstories .details").slideToggle();
            
            if(thisIcon.hasClass("rotated")){
                $(this).removeClass("active");
                $(this).find("span").html(prevBtnText);
                thisIcon.transition({rotate: '0deg'},500).removeClass("rotated");
                $("#ourstories").removeClass("opened", 300);
            }else{
                $(this).addClass("active");
                $(this).find("span").html("Close");   
                thisIcon.transition({rotate: '45deg'},500).addClass("rotated");
                $("#ourstories").addClass("opened", 300);
            }

        });//END of learn more click
  

    function isTouchDevice(){
        return true == ("ontouchstart" in window || window.DocumentTouch && document instanceof DocumentTouch);
    }
        
    
      function gymnastAuto() {

        //used for desktop.. auto play the video when it comes into window 
        //but only do so if not touch screen (Windows)
        var ourstoriesTrigger = $("#ourstories").offset().top;
        var storyVidPlay = false; 
        $(window).scroll(function(){
            var top = window.pageYOffset; 
            //console.log(top);
            
            if(top > ourstoriesTrigger && storyVidPlay === false){
                ourstories_vid.play(); 
                storyVidPlay = true; 
            }

        }); 
      }//end of function gymnastAuto


        function addPlayButton() {

          //add play button.. 
          var playBtn = '<playBtn></playBtn>'; 
          //$('#title_biles').prepend(playBtn); 
          $('playBtn').on("click",function() {
              ourstories_vid.play(); 
              $(this).hide(); //we believe they only want to show once. so we hide play button. could do pause if want... or enable controls...
          });
        }


        function gymnastMobile() {
          
          var mobileVideo = "https://int.nyt.com/data/videotape/finished/2016/09/13/china-desertification/intro_girl_01-toned-1254.mp4";   
              
          if(WURFL.form_factor === "Smartphone"){
              mobileVideo = "https://int.nyt.com/data/videotape/finished/2016/09/13/china-desertification/intro_girl_01-toned-1254.mp4";  
          }
            
          $("figure.bg_video").css({background:"https://static01.nyt.com/subscriptions/Multiproduct/assets_pa2017/images/why/ourstories/china_desert.jpg"});           
          $("figure.bg_video video").attr({poster:"https://static01.nyt.com/subscriptions/Multiproduct/assets_pa2017/images/why/ourstories/china_desert.jpg", src : mobileVideo});
            
         //addPlayButton(); 

        }//END of 


        function handleGymnast() {

          //depends on if device is a mobile device or desktop .. 
          //if not a phone, set trigger to auto play when scrolled to the section .. 
        if(WURFL.form_factor != "Desktop"){
            gymnastMobile();//set up the mobile slider for the IG boxes
            
        } else {
           //set an automator play based on scrolling 
            //gymnastAuto(); 
            //if touch, add the play button in case not scrolling
     


        }
      }

    handleGymnast(); 

  }); 